"""
Step 2: 离线提取 CosyVoice3 FSQ speech token + CAM++ 说话人向量, 生成训练 manifest。

输入 jsonl(每行):
    {"utt": "spk001_0001", "wav": "/abs/path.wav", "text": "今天天气不错。", "speaker": "spk001"}

输出 jsonl(每行, 追加两个字段):
    {..., "codes": [123, 456, ...], "spk_emb": "data/spk_emb/spk001_0001.npy", "dur": 3.42}

用法:
    python train/prepare_data.py \
        --input data/raw.jsonl --output data/train.codes.jsonl \
        --speech_tokenizer pretrained_models/CosyVoice3-0.5B/speech_tokenizer_v3.onnx \
        --campplus pretrained_models/CosyVoice3-0.5B/campplus.onnx \
        --spk_emb_dir data/spk_emb --device cuda

说明:
  - speech tokenizer 提取逻辑镜像 cosyvoice/cli/frontend.py 的 _extract_speech_token
    (whisper 128-mel → onnx)。如果你的 CosyVoice 版本前端有出入, 以 repo 内
    frontend 实现为准, 只要保证和 vocoder(token2wav)用同一套 tokenizer 即可。
  - CAM++ 提取逻辑镜像 pilot_voice/tools/audio.py 的 SpkEmbeddingExtractor
    (kaldi fbank80 + CMN), 输入是 -24dB RMS 归一化后的 16k 音频。
  - 长音频(>30s)对 tokenizer onnx 不友好, 这里直接跳过并记入 skip 日志。
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import whisper
from tqdm import tqdm

TOKEN_RATE = 25.0  # CosyVoice3 speech token 帧率


def load_wav_16k(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav  # (1, T)


def loudness_norm(wav: torch.Tensor) -> torch.Tensor:
    """镜像 pilot_voice.tools.audio.load_audio_norm: RMS 归一到约 -24dB。"""
    gain_db = -24 - math.log10(torch.mean(wav * wav).item() + 1e-16) * 10
    wav = torchaudio.functional.gain(wav, gain_db)
    return torch.clip(wav, min=-1, max=1)


class SpeechTokenizer:
    """CosyVoice3 FSQ speech tokenizer (onnx)。"""

    def __init__(self, onnx_path: str, device: str = "cuda", num_threads: int = 0):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda" else ["CPUExecutionProvider"]
        )
        opt = onnxruntime.SessionOptions()
        opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads > 0:  # 多进程分片时必须限核, 否则每个进程都抢满整机
            opt.intra_op_num_threads = num_threads
        self.session = onnxruntime.InferenceSession(
            onnx_path, sess_options=opt, providers=providers,
        )

    def __call__(self, wav_16k: torch.Tensor) -> list:
        # 镜像 cosyvoice frontend: whisper log-mel(n_mels=128), 不 pad 到 30s
        feat = whisper.log_mel_spectrogram(wav_16k.squeeze(0), n_mels=128)
        tokens = self.session.run(
            None,
            {
                self.session.get_inputs()[0].name: feat.unsqueeze(0).numpy(),
                self.session.get_inputs()[1].name: np.array([feat.shape[1]], dtype=np.int32),
            },
        )[0]
        return tokens.flatten().tolist()


class CampPlusExtractor:
    """CAM++ 192 维说话人向量 (onnx, CPU 足够快)。"""

    def __init__(self, onnx_path: str, num_threads: int = 0):
        opt = onnxruntime.SessionOptions()
        if num_threads > 0:
            opt.intra_op_num_threads = num_threads
        self.session = onnxruntime.InferenceSession(
            onnx_path, sess_options=opt, providers=["CPUExecutionProvider"],
        )

    def __call__(self, wav_16k_norm: torch.Tensor) -> np.ndarray:
        feat = kaldi.fbank(wav_16k_norm, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        emb = self.session.run(
            None, {self.session.get_inputs()[0].name: feat.unsqueeze(0).numpy()},
        )[0]
        return emb.flatten().astype(np.float32)  # (192,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--speech_tokenizer", required=True)
    ap.add_argument("--campplus", required=True)
    ap.add_argument("--spk_emb_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_sec", type=float, default=30.0)
    ap.add_argument("--min_sec", type=float, default=0.5)
    ap.add_argument("--num_threads", type=int, default=0,
                    help="每进程 CPU 线程数(0=onnxruntime 默认, 分片并行时务必设置)")
    args = ap.parse_args()

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    spk_emb_dir = Path(args.spk_emb_dir)
    spk_emb_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = SpeechTokenizer(args.speech_tokenizer, args.device, args.num_threads)
    campplus = CampPlusExtractor(args.campplus, args.num_threads)

    n_ok, n_skip = 0, 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in tqdm(fin, desc="extract"):
            item = json.loads(line)
            try:
                wav = load_wav_16k(item["wav"])
                dur = wav.size(1) / 16000.0
                if not (args.min_sec <= dur <= args.max_sec):
                    n_skip += 1
                    continue

                codes = tokenizer(wav)
                # sanity: token 数应约等于 dur * 25
                if abs(len(codes) - dur * TOKEN_RATE) > 0.2 * dur * TOKEN_RATE + 10:
                    print(f"[warn] {item['utt']}: {len(codes)} tokens vs {dur:.1f}s")

                emb = campplus(loudness_norm(wav))
                emb_path = spk_emb_dir / f"{item['utt']}.npy"
                np.save(emb_path, emb)

                item.update(codes=codes, spk_emb=str(emb_path), dur=round(dur, 3))
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                n_skip += 1
                print(f"[skip] {item.get('utt')}: {e}")

    print(f"done. ok={n_ok} skip={n_skip} -> {args.output}")


if __name__ == "__main__":
    main()
