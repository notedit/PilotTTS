import os
import sys

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


def load_engine(config_path="configs/infer.yaml", checkpoint=None, device=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if checkpoint:
        config["checkpoint_path"] = checkpoint
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return InferenceEngine(config, device)


def synthesize(engine, text, prompt_wav,
               emotion=None, language="zh", output_path="output.wav"):
    """Synthesize speech from text.

    Args:
        engine: InferenceEngine instance.
        text: Text to synthesize.
        prompt_wav: Path to the prompt audio file.
        emotion: Optional emotion tag (e.g. "happy", "sad", "angry", "surprise", "fear").
                 When set, the text will be wrapped with emotion tags internally.
        language: Language tag (e.g. "zh", "en", "zh-tianjin", "zh-sichuan").
        output_path: Path to save the output wav file.
    """
    if emotion:
        text = f"<|{emotion}|>{text}<|/{emotion}|>"

    codes, speech = engine.synthesize(prompt_wav, text, language=language)

    sample_rate = engine.config["vocoder"].get("sample_rate", 24000)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torchaudio.save(output_path, speech.cpu(), sample_rate=sample_rate)
    print(f"Saved: {output_path} ({speech.shape[-1] / sample_rate:.2f}s)")


if __name__ == "__main__":
    prompt_wav = "assert/prompt.wav"

    # ==================== 1. Zero-shot Voice Cloning ====================
    print("=" * 50)
    print("[1] Zero-shot Voice Cloning")
    print("=" * 50)

    engine = load_engine(checkpoint="pretrained_models/pilot_tts.pt")

    synthesize(
        engine,
        text="今天天气真不错，适合出门散步。",
        prompt_wav=prompt_wav,
        output_path="output/demo_zeroshot.wav",
    )

    del engine
    torch.cuda.empty_cache()

    # ==================== 2. Emotion Synthesis (Instruct) ====================
    print("\n" + "=" * 50)
    print("[2] Emotion Synthesis (Instruct)")
    print("=" * 50)

    engine_instruct = load_engine(checkpoint="pretrained_models/pilot_tts_instruct.pt")

    synthesize(
        engine_instruct,
        text="今天天气真好啊，我们去公园玩吧！",
        prompt_wav=prompt_wav,
        emotion="happy",
        output_path="output/demo_emotion_happy.wav",
    )

    synthesize(
        engine_instruct,
        text="我真的好难过，为什么会这样。",
        prompt_wav=prompt_wav,
        emotion="sad",
        output_path="output/demo_emotion_sad.wav",
    )

    # ==================== 3. Paralanguage Synthesis (Instruct) ====================
    print("\n" + "=" * 50)
    print("[3] Paralanguage Synthesis (Instruct)")
    print("=" * 50)

    synthesize(
        engine_instruct,
        text="这个笑话太好笑了<|LAUGH|>我真的忍不住。",
        prompt_wav=prompt_wav,
        output_path="output/demo_paralanguage.wav",
    )

    # ==================== 4. Dialect Synthesis (Instruct) ====================
    print("\n" + "=" * 50)
    print("[4] Dialect Synthesis (Henan)")
    print("=" * 50)

    synthesize(
        engine_instruct,
        text="中不中啊，咱俩一块儿去吃胡辣汤吧。",
        prompt_wav=prompt_wav,
        language="zh-henan",
        output_path="output/demo_dialect_henan.wav",
    )

    del engine_instruct
    torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print("All demos completed. Check output/ directory.")
    print("=" * 50)
