"""
Step 3: 在训练集上计算 w2v-bert-2.0 layer-17 特征的全局 per-dim mean/var。

产出 {"mean": (1024,), "var": (1024,)}, 即 model 配置里的 w2v_stats_path。
这一步必须在你自己的数据分布上做 —— 用别人的 stats 会导致 conformer 输入
scale 漂移, 训练前期 loss 异常。采样 2~5 万条(几百小时)就足够收敛。

用法:
    python train/compute_w2v_stats.py \
        --manifest data/train.codes.jsonl \
        --w2v_path pretrained_models/w2v-bert-2.0 \
        --output pretrained_models/wav2vec2bert_stats.pt \
        --max_utts 30000 --batch_size 16 --device cuda
"""

import argparse
import json
import random

import torch
import torchaudio
from tqdm import tqdm
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

LAYER = 17  # 与 pilot_voice/model.py _get_wav2vec_features 对齐


def load_wav_16k(path):
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0).numpy()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--w2v_path", default="facebook/w2v-bert-2.0")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_utts", type=int, default=30000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    wavs = [json.loads(l)["wav"] for l in open(args.manifest)]
    random.seed(0)
    random.shuffle(wavs)
    wavs = wavs[: args.max_utts]

    extractor = SeamlessM4TFeatureExtractor.from_pretrained(args.w2v_path)
    model = Wav2Vec2BertModel.from_pretrained(args.w2v_path).to(args.device).eval()

    # sum / sumsq 双精度累积, 数值上足够稳
    s = torch.zeros(1024, dtype=torch.float64, device=args.device)
    sq = torch.zeros(1024, dtype=torch.float64, device=args.device)
    n = 0

    for i in tqdm(range(0, len(wavs), args.batch_size), desc="stats"):
        batch = [load_wav_16k(p) for p in wavs[i : i + args.batch_size]]
        inputs = extractor(batch, sampling_rate=16000, return_tensors="pt")
        feats = inputs["input_features"].to(args.device)
        attn = inputs["attention_mask"].to(args.device)
        h = model(
            input_features=feats, attention_mask=attn, output_hidden_states=True,
        ).hidden_states[LAYER]                       # (B, T, 1024)
        m = attn.bool()
        x = h[m].double()                            # (N_valid, 1024)
        s += x.sum(0)
        sq += (x * x).sum(0)
        n += x.size(0)

    mean = (s / n).float().cpu()
    var = (sq / n - (s / n) ** 2).float().cpu().clamp_min(1e-8)
    torch.save({"mean": mean, "var": var}, args.output)
    print(f"saved {args.output}  frames={n}  mean|.|={mean.abs().mean():.4f}  var~={var.mean():.4f}")


if __name__ == "__main__":
    main()
