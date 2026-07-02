"""
正式训练前的 sanity check: 取 manifest 前 8 条跑一个完整 forward/backward,
验证序列构造、mask 对齐、conds 注入、loss 数值(初始应 ≈ ln(151936) ≈ 11.93,
lm_head 是 xavier 重新初始化的, CE 在完整 lm_head 输出维度上算,
推理时才把 logits 截到前 audio_tokens=6563 维)。

    python train/sanity_check.py --config train/config.yaml
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pilot_voice.model import PilotVoice
from pilot_voice.utils import build_semantic_model
from train.dataset import Collator, PilotTTSDataset
from train.train import compute_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    mcfg, dcfg = cfg["model"], cfg["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = PilotTTSDataset(
        dcfg["train_manifest"], cfg["tokenizer"]["path"],
        language=cfg["language"], vocab_size=mcfg["vocab_size"],
    )
    collate = Collator(mcfg["w2v_path"])
    batch = collate([ds[i] for i in range(min(8, len(ds)))])

    sample = ds[0]
    seq = sample["seq"]
    print(f"[seq] len={seq.size(0)} text_len(incl BOA)={sample['text_len']} "
          f"target_len(incl EOS)={sample['target_len']}")
    print(f"[seq] head={seq[:8].tolist()} (应全部 >= 6563)")
    b = sample["text_len"] - 1
    print(f"[seq] boa={seq[b].item()} (应=6561)  eos={seq[-1].item()} (应=6562)")
    codes = seq[b + 1 : -1]
    assert codes.min() >= 0 and codes.max() < mcfg["vocab_size"], "codes 越界"

    model = PilotVoice(
        audio_tokens=mcfg["audio_tokens"], pretrain_path=mcfg["pretrain_path"],
        w2v_path=mcfg["w2v_path"], w2v_stats_path=mcfg["w2v_stats_path"],
        use_conditioning=True, use_spk_emb=mcfg["use_spk_emb"],
        init_from_pretrained=True,
    ).to(device)

    semantic_model, sm_mean, sm_std = build_semantic_model(
        mcfg["w2v_stats_path"], w2v_path=mcfg["w2v_path"],
    )
    semantic_model.to(device).eval()

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        loss, acc = compute_loss(
            model, batch, device, cfg, semantic_model, sm_mean, sm_std, drop_cond=False,
        )
    loss.backward()

    expect = math.log(151936)  # CE 在完整 lm_head 输出维度(Qwen3 词表)上算
    print(f"[loss] {loss.item():.4f} (随机初始化 lm_head 期望 ≈ {expect:.2f})  acc={acc.item():.4f}")

    g_perc = sum(
        p.grad.abs().sum().item()
        for n, p in model.named_parameters()
        if p.grad is not None and "perceiver" in n
    )
    g_conf = sum(
        p.grad.abs().sum().item()
        for n, p in model.named_parameters()
        if p.grad is not None and "conditioning_encoder" in n
    )
    print(f"[grad] perceiver |g|={g_perc:.3e}  conformer |g|={g_conf:.3e} (都应 > 0, "
          f"说明 conds 前缀在计算图里)")
    print("sanity check OK")


if __name__ == "__main__":
    main()
