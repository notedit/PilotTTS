"""
PilotTTS 训练主脚本(单机多卡, torchrun)。

    # stage 1: 冻结 LLM 主干, 只训 conditioner + embed_tokens + lm_head
    torchrun --nproc_per_node=8 train/train.py --config train/config.yaml

    # stage 2: 改 config 里 stage=2, lr=2e-5, resume=stage1 ckpt, save_dir 换新
    torchrun --nproc_per_node=8 train/train.py --config train/config.yaml

关键实现点(与 pilot_voice 推理侧严格对齐):
  1. 直接实例化 pilot_voice.model.PilotVoice —— 保证 state_dict key 与
     engine.py 的 strict=True 加载完全兼容。
  2. w2v-bert 前端 *不* 挂在模型上(不调 load_semantic_model), 而是独立
     实例化 —— 否则 ar.semantic_model.* 会进 state_dict, engine 加载直接报错。
  3. loss 只在 audio 段(codes + EOS)计算, 见 dataset.Collator 注释。
  4. cond_dropout: 按 batch 概率丢弃 conditioning 前缀(prompt_audio=None,
     spk_emb=None), 提升鲁棒性; 需要 DDP find_unused_parameters=True。
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo 根

from pilot_voice.model import PilotVoice                     # noqa: E402
from pilot_voice.utils import build_semantic_model           # noqa: E402
from train.dataset import Collator, PilotTTSDataset          # noqa: E402


# ----------------------------------------------------------------------
def setup_dist():
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return rank, dist.get_world_size()
    return 0, 1


def is_main(rank):
    return rank == 0


def lr_lambda(step, warmup, max_steps):
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, max_steps - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def set_stage_trainable(model: PilotVoice, stage: int):
    """stage1: conditioner + embed_tokens + lm_head; stage2: 全量。"""
    if stage >= 2:
        for p in model.parameters():
            p.requires_grad = True
        return
    for name, p in model.named_parameters():
        p.requires_grad = (
            name.startswith("ar.conditioning_encoder")
            or name.startswith("ar.perceiver_encoder")
            or "embed_tokens" in name
            or "lm_head" in name
        )


def engine_compatible_state_dict(model):
    """去 DDP 前缀 + 过滤任何 semantic_model 残留, 保证 engine strict=True 可加载。"""
    sd = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    return {
        k: v
        for k, v in sd.items()
        if not k.startswith(("ar.semantic_model", "ar.semantic_mean", "ar.semantic_std"))
    }


# ----------------------------------------------------------------------
@torch.no_grad()
def get_w2v_features(semantic_model, mean, std, input_features, attention_mask):
    """镜像 AR._get_wav2vec_features: layer-17 + 全局归一化。"""
    out = semantic_model(
        input_features=input_features,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    feat = out.hidden_states[17]
    return (feat - mean.to(feat.device)) / std.to(feat.device)


def compute_loss(model, batch, device, cfg, semantic_model, sm_mean, sm_std, drop_cond):
    all_token = batch["all_token"].to(device, non_blocking=True)
    loss_mask = batch["loss_mask"].to(device, non_blocking=True)
    text_len = batch["text_len"].to(device)
    target_len = batch["target_len"].to(device)

    if drop_cond:
        prompt_audio, spk_emb = None, None
    else:
        w2v_feat = batch["w2v_input_features"].to(device, non_blocking=True)
        w2v_mask = batch["w2v_attention_mask"].to(device, non_blocking=True)
        spk_cond_emb = get_w2v_features(semantic_model, sm_mean, sm_std, w2v_feat, w2v_mask)
        prompt_audio = {"spk_cond_emb": spk_cond_emb, "attention_mask": w2v_mask}
        spk_emb = batch["spk_emb"].to(device) if cfg["model"]["use_spk_emb"] else None

    # PilotVoice.forward -> AR.forward:
    #   实际只用 all_token / text_len / target_len / spk_emb / prompt_audio
    out_ar, target_ar = model(
        text=None, text_len=text_len, target=None, target_len=target_len,
        proms_len=None, all_token_list=all_token,
        spk_emb=spk_emb, prompt_audio=prompt_audio,
    )
    # out_ar: (B, V, T-1) 已按 conds 长度 offset 对齐; target_ar: (B, T-1)
    # 文本段 target id ≥ 6563, 超出 lm_head 输出维度(Qwen 词表 151936),
    # cross_entropy 对每个位置都做类别越界校验, 必须在调用前用 ignore_index
    # 屏蔽非 audio 段 —— 事后乘 loss_mask 来不及, CUDA 上直接 device assert。
    target_ce = target_ar.long().masked_fill(~loss_mask, -100)
    ce = F.cross_entropy(out_ar, target_ce, reduction="none", ignore_index=-100)
    loss = ce.sum() / loss_mask.sum().clamp_min(1)

    with torch.no_grad():
        acc = ((out_ar.argmax(1) == target_ce) & loss_mask).sum() / loss_mask.sum().clamp_min(1)
    return loss, acc


@torch.no_grad()
def validate(
    model, val_loader, device, cfg,
    semantic_model, sm_mean, sm_std, autocast, world, max_batches=50,
):
    """所有 rank 同步执行(DDP forward 含集合通信, 只在 rank0 跑会挂死),
    各 rank 消费 DistributedSampler 切好的不同分片, 指标 all_reduce 求均值。"""
    model.eval()
    vl, va, nb = 0.0, 0.0, 0
    with autocast:
        for vb in val_loader:
            l, a = compute_loss(
                model, vb, device, cfg,
                semantic_model, sm_mean, sm_std, False,
            )
            vl += l.item(); va += a.item(); nb += 1
            if nb >= max_batches:
                break
    model.train()
    stats = torch.tensor([vl, va, float(nb)], device=device, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(stats)
    vl, va, nb = stats.tolist()
    return vl / max(nb, 1.0), va / max(nb, 1.0)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    tcfg, mcfg, dcfg = cfg["train"], cfg["model"], cfg["data"]

    rank, world = setup_dist()
    device = torch.device(f"cuda:{rank % max(1, torch.cuda.device_count())}")
    torch.manual_seed(tcfg["seed"] + rank)

    # ---------- data ----------
    ds = PilotTTSDataset(
        dcfg["train_manifest"], cfg["tokenizer"]["path"],
        language=cfg["language"], vocab_size=mcfg["vocab_size"],
        min_target_sec=dcfg["min_target_sec"], max_target_sec=dcfg["max_target_sec"],
        prompt_min_sec=dcfg["prompt_min_sec"], prompt_max_sec=dcfg["prompt_max_sec"],
    )
    val_ds = PilotTTSDataset(
        dcfg["val_manifest"], cfg["tokenizer"]["path"],
        language=cfg["language"], vocab_size=mcfg["vocab_size"],
    )
    collate = Collator(mcfg["w2v_path"])
    sampler = DistributedSampler(ds) if world > 1 else None
    loader = DataLoader(
        ds, batch_size=tcfg["batch_size"], sampler=sampler, shuffle=sampler is None,
        num_workers=dcfg["num_workers"], collate_fn=collate,
        pin_memory=True, drop_last=True, persistent_workers=dcfg["num_workers"] > 0,
    )
    # val 也按 rank 切分(shuffle=False, 不足时 DistributedSampler 自动补齐,
    # 保证各 rank batch 数一致, validate 里的集合通信不会错位)
    val_sampler = DistributedSampler(val_ds, shuffle=False) if world > 1 else None
    val_loader = DataLoader(
        val_ds, batch_size=tcfg["batch_size"], sampler=val_sampler, shuffle=False,
        num_workers=2, collate_fn=collate,
    )

    # ---------- model ----------
    model = PilotVoice(
        audio_tokens=mcfg["audio_tokens"],
        pretrain_path=mcfg["pretrain_path"],
        w2v_path=mcfg["w2v_path"],
        w2v_stats_path=mcfg["w2v_stats_path"],
        use_conditioning=mcfg["use_conditioning"],
        use_spk_emb=mcfg["use_spk_emb"],
        init_from_pretrained=True,   # 训练必须加载 Qwen3 权重
    )
    set_stage_trainable(model, tcfg["stage"])
    if tcfg["gradient_checkpointing"]:
        model.ar.ar_decoder.gradient_checkpointing_enable()
        model.ar.ar_decoder.config.use_cache = False
    model.to(device)

    # w2v 前端独立实例化(冻结, 不进 DDP / 不进 state_dict)
    semantic_model, sm_mean, sm_std = build_semantic_model(
        mcfg["w2v_stats_path"], w2v_path=mcfg["w2v_path"],
    )
    semantic_model.to(device).eval()
    for p in semantic_model.parameters():
        p.requires_grad = False

    if world > 1:
        model = DDP(
            model, device_ids=[device.index],
            find_unused_parameters=(tcfg["cond_dropout"] > 0 or tcfg["stage"] == 1),
        )

    params = [p for p in model.parameters() if p.requires_grad]
    if is_main(rank):
        n_train = sum(p.numel() for p in params)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"[model] trainable {n_train/1e6:.1f}M / {n_total/1e6:.1f}M  stage={tcfg['stage']}")

    opt = torch.optim.AdamW(
        params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"],
        betas=tuple(tcfg["betas"]),
    )
    # scheduler 每 grad_accum 个 micro-step 才走 1 步(optimizer step 口径),
    # warmup/cosine 地平线必须换算, 否则 cosine 只退火 1/grad_accum 就结束训练
    opt_max_steps = max(1, tcfg["max_steps"] // tcfg["grad_accum"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, tcfg["warmup_steps"], opt_max_steps),
    )

    step = 0
    if tcfg["resume"]:
        ckpt = torch.load(tcfg["resume"], map_location="cpu")
        target = model.module if hasattr(model, "module") else model
        missing, unexpected = target.load_state_dict(ckpt["model"], strict=False)
        assert not unexpected, unexpected
        if "optimizer" in ckpt and ckpt.get("stage") == tcfg["stage"]:
            opt.load_state_dict(ckpt["optimizer"])
            sched.load_state_dict(ckpt["scheduler"])
            step = ckpt["step"]
        if is_main(rank):
            print(f"[resume] {tcfg['resume']} step={step} missing={len(missing)}")

    save_dir = Path(tcfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=tcfg["bf16"])

    # ---------- loop ----------
    model.train()
    t0, run_loss, run_acc = time.time(), 0.0, 0.0
    epoch = 0
    done = False
    while not done:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            drop_cond = torch.rand(1).item() < tcfg["cond_dropout"]
            if world > 1:  # 各 rank 保持一致, 避免 DDP unused-param 不同步
                flag = torch.tensor([drop_cond], device=device, dtype=torch.uint8)
                dist.broadcast(flag, src=0)
                drop_cond = bool(flag.item())

            with autocast:
                loss, acc = compute_loss(
                    model, batch, device, cfg,
                    semantic_model, sm_mean, sm_std, drop_cond,
                )
            (loss / tcfg["grad_accum"]).backward()
            run_loss += loss.item()
            run_acc += acc.item()

            if (step + 1) % tcfg["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(params, tcfg["grad_clip"])
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

            step += 1
            if is_main(rank) and step % tcfg["log_interval"] == 0:
                dt = time.time() - t0
                print(
                    f"step {step} | loss {run_loss/tcfg['log_interval']:.4f} "
                    f"| acc {run_acc/tcfg['log_interval']:.4f} "
                    f"| lr {sched.get_last_lr()[0]:.2e} "
                    f"| {tcfg['log_interval']/dt:.2f} it/s"
                )
                t0, run_loss, run_acc = time.time(), 0.0, 0.0

            if step % tcfg["val_interval"] == 0:
                vloss, vacc = validate(
                    model, val_loader, device, cfg,
                    semantic_model, sm_mean, sm_std, autocast, world,
                )
                if is_main(rank):
                    print(f"[val] step {step} | loss {vloss:.4f} | acc {vacc:.4f}")

            if is_main(rank) and step % tcfg["save_interval"] == 0:
                path = save_dir / f"step_{step}.pt"
                torch.save(
                    {
                        "model": engine_compatible_state_dict(model),
                        "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(),
                        "step": step,
                        "stage": tcfg["stage"],
                        "config": cfg,
                    },
                    path,
                )
                print(f"[save] {path}")

            if step >= tcfg["max_steps"]:
                done = True
                break
        epoch += 1

    if is_main(rank):
        torch.save({"model": engine_compatible_state_dict(model), "step": step,
                    "stage": tcfg["stage"], "config": cfg}, save_dir / "final.pt")
        print(f"[done] {save_dir/'final.pt'}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
