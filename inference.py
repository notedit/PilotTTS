import os
import sys
import argparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
for p in [
    os.path.join(ROOT_DIR, "third_party"),
    os.path.join(ROOT_DIR, "third_party", "Matcha-TTS"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml
import torch
import torchaudio

from pilot_voice.engine import InferenceEngine


def main():
    parser = argparse.ArgumentParser(description="PilotVoice Inference")
    parser.add_argument("--config", default="configs/infer.yaml", help="YAML config file")
    parser.add_argument("--checkpoint", default="pretrained_models/pilot_tts.pt",
                        help="Model checkpoint path")
    parser.add_argument("--prompt-wav", required=True, help="Prompt audio path")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--emotion", default=None,
                        help="Emotion tag (happy, sad, angry, surprise, fear, etc.)")
    parser.add_argument("--language", default="zh", help="Language tag (zh, en, zh-tianjin, etc.)")
    parser.add_argument("--output", default="output.wav", help="Output wav path")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["checkpoint_path"] = args.checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = InferenceEngine(config, device)

    text = args.text
    if args.emotion:
        text = f"<|{args.emotion}|>{text}<|/{args.emotion}|>"

    codes, speech = engine.synthesize(
        args.prompt_wav, text,
        language=args.language,
    )

    sample_rate = config["vocoder"].get("sample_rate", 24000)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torchaudio.save(args.output, speech.cpu(), sample_rate=sample_rate)
    print(f"Saved to {args.output} ({speech.shape[-1] / sample_rate:.2f}s)")


if __name__ == "__main__":
    main()
