"""
Step 1(Emilia 版): Emilia-Dataset WebDataset tar → PilotTTS 训练 raw jsonl。

从 HF 下载 Emilia/ZH(CC BY-NC)或 Emilia-YODAS/ZH(CC BY 4.0, 可商用)的
tar 分片, 解出 mp3 + json, 过滤后写成 prepare_data.py 需要的格式:

    {"utt": "ZH_B00000_S06226_W000000", "wav": "/abs/path.mp3",
     "text": "...", "speaker": "ZH_B00000_S06226", "dur": 4.3, "dnsmos": 3.4}

并按 speaker 切 train/val(val 全部是 unseen speaker, 才能测 zero-shot)。

用法:
    python train/emilia_to_jsonl.py \
        --subset Emilia/ZH --num_tars 4 \
        --out_dir data/emilia \
        --min_sec 2.0 --max_sec 30.0 --min_utts_per_speaker 2 \
        --val_speaker_ratio 0.02 --workers 4

说明:
  - HF token 从 ~/.cache/huggingface/token 读取(Emilia 是 gated 数据集)。
  - mp3 不转码, 训练侧 load_wav_16k 会自动解码重采样。
  - tar 下载有断点续传; --delete_tar 可在解包后删 tar 省盘。
  - Emilia 的 speaker id 是源音频内 diarization 结果, 跨源文件不互通,
    对本 pipeline(只需同 id 内采样另一条做 prompt)足够。
"""

import argparse
import json
import random
import tarfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "amphion/Emilia-Dataset"


def list_subset_tars(subset: str):
    files = list_repo_files(REPO_ID, repo_type="dataset")
    tars = sorted(f for f in files if f.startswith(subset + "/") and f.endswith(".tar"))
    if not tars:
        raise SystemExit(f"no tar found under {subset!r}; e.g. Emilia/ZH, Emilia-YODAS/ZH")
    return tars


def process_one_tar(tar_name: str, audio_root: Path, meta_dir: Path, args):
    """下载 + 解包一个 tar, 写出该 tar 的 per-shard jsonl, 返回 (tar_name, n_ok, n_skip)。"""
    shard_jsonl = meta_dir / (Path(tar_name).stem + ".jsonl")
    if shard_jsonl.exists() and not args.overwrite:
        n = sum(1 for _ in open(shard_jsonl))
        return tar_name, n, -1  # -1 = skipped(已处理过)

    tar_path = hf_hub_download(
        REPO_ID, tar_name, repo_type="dataset",
        local_dir=args.tar_dir, local_dir_use_symlinks=False,
    )

    out_dir = audio_root / Path(tar_name).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok, n_skip = 0, 0
    items = []
    with tarfile.open(tar_path) as tf:
        # 同名 .mp3/.json 成对出现; 先收 json 再解对应 mp3
        metas = {}
        for m in tf.getmembers():
            if m.name.endswith(".json"):
                metas[Path(m.name).stem] = json.load(tf.extractfile(m))
        for m in tf.getmembers():
            if not m.name.endswith(".mp3"):
                continue
            stem = Path(m.name).stem
            meta = metas.get(stem)
            if meta is None:
                n_skip += 1
                continue
            dur = float(meta.get("duration", 0))
            dnsmos = float(meta.get("dnsmos", 0))
            text = (meta.get("text") or "").strip()
            if not (args.min_sec <= dur <= args.max_sec) or dnsmos < args.min_dnsmos or not text:
                n_skip += 1
                continue
            mp3_path = out_dir / f"{stem}.mp3"
            if not mp3_path.exists():
                with open(mp3_path, "wb") as f:
                    f.write(tf.extractfile(m).read())
            items.append({
                "utt": meta.get("id", stem),
                "wav": str(mp3_path.resolve()),
                "text": text,
                "speaker": meta.get("speaker", "_".join(stem.split("_")[:-1])),
                "dur": round(dur, 3),
                "dnsmos": round(dnsmos, 3),
            })
            n_ok += 1

    with open(shard_jsonl, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    if args.delete_tar:
        Path(tar_path).unlink(missing_ok=True)
    return tar_name, n_ok, n_skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="Emilia/ZH", help="Emilia/ZH 或 Emilia-YODAS/ZH 等")
    ap.add_argument("--num_tars", type=int, default=4, help="处理前 N 个 tar; -1 = 全部")
    ap.add_argument("--tar_names", nargs="*", default=None, help="显式指定 tar(覆盖 num_tars)")
    ap.add_argument("--out_dir", default="data/emilia")
    ap.add_argument("--tar_dir", default=None, help="tar 下载目录, 默认 <out_dir>/tars")
    ap.add_argument("--min_sec", type=float, default=2.0)
    ap.add_argument("--max_sec", type=float, default=30.0)
    ap.add_argument("--min_dnsmos", type=float, default=3.0)
    ap.add_argument("--min_utts_per_speaker", type=int, default=2)
    ap.add_argument("--val_speaker_ratio", type=float, default=0.02)
    ap.add_argument("--min_val_speakers", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--delete_tar", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    audio_root = out_dir / "audio"
    meta_dir = out_dir / "shards"
    args.tar_dir = args.tar_dir or str(out_dir / "tars")
    for d in (audio_root, meta_dir, Path(args.tar_dir)):
        d.mkdir(parents=True, exist_ok=True)

    if args.tar_names:
        tars = args.tar_names
    else:
        tars = list_subset_tars(args.subset)
        print(f"[list] {args.subset}: {len(tars)} tars total")
        if args.num_tars > 0:
            tars = tars[: args.num_tars]
    print(f"[plan] processing {len(tars)} tars -> {out_dir}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one_tar, t, audio_root, meta_dir, args): t for t in tars}
        for fut in as_completed(futs):
            tar_name, n_ok, n_skip = fut.result()
            tag = "cached" if n_skip == -1 else f"ok={n_ok} skip={n_skip}"
            print(f"[tar] {tar_name}: {tag}")

    # ---- 合并本次处理的 shard, 按 speaker 过滤 + 切分 ----
    items = []
    for t in tars:
        shard = meta_dir / (Path(t).stem + ".jsonl")
        items += [json.loads(l) for l in open(shard)]

    spk2items = defaultdict(list)
    for it in items:
        spk2items[it["speaker"]].append(it)
    n_before = len(items)
    spk2items = {
        s: its for s, its in spk2items.items() if len(its) >= args.min_utts_per_speaker
    }
    n_after = sum(len(v) for v in spk2items.values())

    speakers = sorted(spk2items)
    random.seed(args.seed)
    random.shuffle(speakers)
    n_val = max(args.min_val_speakers, int(len(speakers) * args.val_speaker_ratio))
    n_val = min(n_val, max(1, len(speakers) // 10))  # val 不超过说话人的 10%
    val_spk = set(speakers[:n_val])

    tr_path, va_path = out_dir / "train.raw.jsonl", out_dir / "val.raw.jsonl"
    n_tr = n_va = h_tr = h_va = 0
    with open(tr_path, "w") as ftr, open(va_path, "w") as fva:
        for s in sorted(spk2items):
            for it in spk2items[s]:
                line = json.dumps(it, ensure_ascii=False) + "\n"
                if s in val_spk:
                    fva.write(line); n_va += 1; h_va += it["dur"]
                else:
                    ftr.write(line); n_tr += 1; h_tr += it["dur"]

    print(
        f"[filter] {n_before} -> {n_after} utts "
        f"(dropped speakers with < {args.min_utts_per_speaker} utts)\n"
        f"[split] train: {n_tr} utts / {h_tr/3600:.1f} h / {len(speakers)-n_val} speakers -> {tr_path}\n"
        f"[split] val:   {n_va} utts / {h_va/3600:.1f} h / {n_val} speakers (unseen) -> {va_path}"
    )


if __name__ == "__main__":
    main()
