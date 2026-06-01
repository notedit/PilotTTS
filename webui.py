import os
import sys
import argparse
import tempfile
import traceback

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
for p in [
    os.path.join(ROOT_DIR, "third_party"),
    os.path.join(ROOT_DIR, "third_party", "Matcha-TTS"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml
import torch
import numpy as np
import gradio as gr

from pilot_voice.engine import InferenceEngine

UI_ONLY = False
ENGINE = None


def load_engine(config_path, checkpoint_path):
    global ENGINE
    if UI_ONLY:
        print("[UI_ONLY] Skipping model loading")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if checkpoint_path:
        config["checkpoint_path"] = checkpoint_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ENGINE = InferenceEngine(config, device)
    print("Model loaded successfully.")


def tts_zero_shot(prompt_audio, target_text):
    if not target_text.strip():
        raise gr.Error("Please enter target text.")
    if prompt_audio is None:
        raise gr.Error("Please upload prompt audio.")

    if UI_ONLY:
        return (24000, np.zeros(24000, dtype=np.float32))

    try:
        codes, speech = ENGINE.synthesize(
            prompt_audio,
            target_text.strip(),
        )
        wav = speech.squeeze().cpu().numpy()
        return (24000, wav)
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Synthesis failed: {e}")


def tts_instruct(prompt_audio, target_text, emotion, language):
    if not target_text.strip():
        raise gr.Error("Please enter target text.")
    if prompt_audio is None:
        raise gr.Error("Please upload prompt audio.")

    if UI_ONLY:
        return (24000, np.zeros(24000, dtype=np.float32))

    text = target_text.strip()
    emotion = emotion.strip() if emotion else ""
    language = language.strip() if language else ""

    if emotion:
        text = f"<|{emotion}|>{text}<|/{emotion}|>"

    lang_param = language if language else None

    try:
        codes, speech = ENGINE.synthesize(
            prompt_audio,
            text,
            language=lang_param,
        )
        wav = speech.squeeze().cpu().numpy()
        return (24000, wav)
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Synthesis failed: {e}")


with gr.Blocks(css="""
body {background-color: #f5f7fb;}
.gradio-container {max-width: 1100px; margin: 0 auto;}
""") as demo:

    gr.Markdown(
        """
        <div style="text-align:center; margin-top: 20px; margin-bottom: 10px;">
          <h1>PilotVoice TTS</h1>
          <p style="color: #666;">
            Upload a prompt speech and text, then generate target speech.
            Supports zero-shot cloning and instruct-based emotion/language control.
          </p>
        </div>
        """
    )

    with gr.Accordion("Quick Start Guide", open=False):
        gr.Markdown(
            """
**Zero-shot TTS (Voice Cloning):**
1. Upload a reference audio (prompt speech), 5-15s recommended, clean single-speaker.
2. Enter the text you want to synthesize in "Target Text".
3. Click "Generate".

**Instruct TTS (Emotion / Paralanguage / Dialect):**
- Same as zero-shot, plus you can control emotion and language/dialect.
- **Emotion**: Fill in the emotion field (e.g. `happy`).
- **Language/Dialect**: Fill in the language field (e.g. `zh-tianjin`) to control dialect.
- **Paralanguage**: You can insert paralanguage tags directly in the target text, e.g. `这个笑话太好笑了<|LAUGH|>我真的忍不住`.
"""
        )

    with gr.Tabs():
        with gr.Tab("Zero-shot TTS"):
            with gr.Row():
                with gr.Column(scale=2):
                    zs_prompt_audio = gr.Audio(
                        label="Prompt Audio",
                        type="filepath",
                    )
                    zs_target_text = gr.Textbox(
                        label="Target Text",
                        placeholder="Text to synthesize",
                        lines=4,
                    )
                    zs_btn = gr.Button("Generate", variant="primary")

                with gr.Column(scale=2):
                    zs_output = gr.Audio(label="Output Audio", format="wav")

            zs_btn.click(
                tts_zero_shot,
                inputs=[zs_prompt_audio, zs_target_text],
                outputs=zs_output,
            )

        with gr.Tab("Instruct TTS"):
            with gr.Row():
                with gr.Column(scale=2):
                    ins_prompt_audio = gr.Audio(
                        label="Prompt Audio",
                        type="filepath",
                    )
                    ins_target_text = gr.Textbox(
                        label="Target Text",
                        placeholder="Text to synthesize",
                        lines=4,
                    )
                    with gr.Row():
                        ins_emotion = gr.Textbox(
                            label="Emotion",
                            placeholder="e.g. happy, sad, angry",
                            lines=1,
                        )
                        ins_language = gr.Textbox(
                            label="Language",
                            placeholder="e.g. zh, en, zh-sichuan",
                            lines=1,
                        )
                    ins_btn = gr.Button("Generate", variant="primary")

                with gr.Column(scale=2):
                    ins_output = gr.Audio(label="Output Audio", format="wav")

            ins_btn.click(
                tts_instruct,
                inputs=[
                    ins_prompt_audio, ins_target_text,
                    ins_emotion, ins_language,
                ],
                outputs=ins_output,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PilotVoice WebUI")
    parser.add_argument("--config", default="configs/infer_pilot_tts.yaml")

    parser.add_argument("--checkpoint", default="pretrained_models/pilot_tts.pt")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--ui-only", action="store_true",
                        help="Launch UI without loading models (for frontend debugging)")
    args = parser.parse_args()

    if args.ui_only:
        UI_ONLY = True
        print("== UI_ONLY mode: no model loaded ==")
    else:
        load_engine(args.config, args.checkpoint)

    demo.queue(max_size=10, default_concurrency_limit=2).launch(
        server_name="0.0.0.0",
        server_port=args.port,
    )
