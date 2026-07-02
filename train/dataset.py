"""
Dataset / collate: 严格按 PilotTTS 推理协议构造训练样本。

序列布局(见 pilot_voice/engine.py ar_inference):
    [ text_ids + vocab_size + 2 ] [ BOA=vocab_size ] [ codes ] [ EOS=vocab_size+1 ]
      文本段(id >= 6563)             6561              0..6560     6562

conditioning prompt 的选择是训练侧最关键的复现细节:
    w2v-bert layer-17 是语义特征 —— 如果 prompt 用目标音频本身, conditioner
    会直接泄漏内容, 模型学会"抄答案", 推理时(prompt 是参考音色而非目标内容)
    立刻崩。所以必须用 *同说话人的另一条 utterance*; 单条说话人退化为
    非重叠随机裁剪(仍有轻微泄漏风险, 数据里这类说话人越少越好)。

spk_emb(CAM++)同样取自 prompt utterance, 与推理侧(都来自参考音频)对齐。
"""

import json
import math
import random
from collections import defaultdict

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import AutoTokenizer


# --- 镜像 pilot_voice/tools/text.py, 避免 import 链拖进 cosyvoice 依赖 ---
def format_text_input(prompt_text, text, include_prompt=False, language="zh"):
    text = text.strip().replace("#3", ",")
    body = f"<|{language}|>{text}"
    return f"<|0.00|>{body}<|0.02|>"


def loudness_norm(wav: torch.Tensor) -> torch.Tensor:
    gain_db = -24 - math.log10(torch.mean(wav * wav).item() + 1e-16) * 10
    wav = torchaudio.functional.gain(wav, gain_db)
    return torch.clip(wav, min=-1, max=1)


def load_wav_16k(path):
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav  # (1, T)


class PilotTTSDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        tokenizer_path: str,
        language: str = "zh",
        vocab_size: int = 6561,
        min_target_sec: float = 0.5,
        max_target_sec: float = 30.0,
        prompt_min_sec: float = 2.0,
        prompt_max_sec: float = 15.0,
        token_rate: float = 25.0,
    ):
        self.vocab_size = vocab_size
        self.boa = vocab_size          # 6561
        self.eos = vocab_size + 1      # 6562
        self.text_offset = vocab_size + 2  # 6563
        self.language = language
        self.prompt_min_sec = prompt_min_sec
        self.prompt_max_sec = prompt_max_sec

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self.items = []
        for line in open(manifest):
            it = json.loads(line)
            n = len(it["codes"])
            if min_target_sec * token_rate <= n <= max_target_sec * token_rate:
                self.items.append(it)

        # speaker -> item indices, 用于采样 conditioning prompt
        self.spk2idx = defaultdict(list)
        for i, it in enumerate(self.items):
            self.spk2idx[it["speaker"]].append(i)

        n_single = sum(1 for v in self.spk2idx.values() if len(v) == 1)
        print(
            f"[dataset] {len(self.items)} utts, {len(self.spk2idx)} speakers, "
            f"{n_single} single-utt speakers (prompt 将退化为自身裁剪)"
        )

    def __len__(self):
        return len(self.items)

    def _sample_prompt(self, idx):
        """同说话人异 utterance 优先; 否则自身随机裁剪。返回 (wav(1,T), spk_emb(192,))"""
        item = self.items[idx]
        cand = self.spk2idx[item["speaker"]]
        if len(cand) > 1:
            j = idx
            while j == idx:
                j = random.choice(cand)
            p = self.items[j]
        else:
            p = item

        wav = load_wav_16k(p["wav"])
        max_len = int(self.prompt_max_sec * 16000)
        min_len = int(self.prompt_min_sec * 16000)
        if wav.size(1) > max_len:
            # 随机裁剪一个 [min, max] 秒的窗口
            win = random.randint(min_len, max_len)
            start = random.randint(0, wav.size(1) - win)
            wav = wav[:, start : start + win]

        spk_emb = torch.from_numpy(np.load(p["spk_emb"]).astype(np.float32))  # (192,)
        return wav, spk_emb

    def __getitem__(self, idx):
        item = self.items[idx]

        text_inp = format_text_input("", item["text"], language=self.language)
        text_ids = self.tokenizer.encode(text_inp, add_special_tokens=False)

        seq = (
            [t + self.text_offset for t in text_ids]
            + [self.boa]
            + item["codes"]
            + [self.eos]
        )
        prompt_wav, spk_emb = self._sample_prompt(idx)

        return {
            "seq": torch.tensor(seq, dtype=torch.long),
            # text_len 含 BOA, target_len 含 EOS —— 与推理侧
            # prompt_lengths = text_len + proms_p_len(1) 的口径一致
            "text_len": len(text_ids) + 1,
            "target_len": len(item["codes"]) + 1,
            "prompt_wav": prompt_wav.squeeze(0).numpy(),  # 16k float32
            "spk_emb": spk_emb,
        }


class Collator:
    """pad + loss mask + w2v 特征提取(mel 在 CPU dataloader worker 里做)。"""

    def __init__(self, w2v_path: str):
        from transformers import SeamlessM4TFeatureExtractor

        self.extractor = SeamlessM4TFeatureExtractor.from_pretrained(w2v_path)

    def __call__(self, batch):
        B = len(batch)
        lens = [b["seq"].size(0) for b in batch]
        T = max(lens)

        all_token = torch.zeros(B, T, dtype=torch.long)
        text_len = torch.tensor([b["text_len"] for b in batch], dtype=torch.long)
        target_len = torch.tensor([b["target_len"] for b in batch], dtype=torch.long)

        # loss_mask: (B, T-1), 对齐 target = all_token[:, 1:]
        # 位置 j 预测 token j+1; 只在 audio 段算 loss:
        #   text_len <= j+1 < text_len + target_len
        # (BOA 是确定性分隔符, 不算; 文本 id 超出 lm_head 输出维度, 架构上不可算)
        loss_mask = torch.zeros(B, T - 1, dtype=torch.bool)
        for i, b in enumerate(batch):
            all_token[i, : lens[i]] = b["seq"]
            lo = b["text_len"] - 1
            hi = b["text_len"] + b["target_len"] - 1
            loss_mask[i, lo:hi] = True

        inputs = self.extractor(
            [b["prompt_wav"] for b in batch],
            sampling_rate=16000,
            return_tensors="pt",
        )

        spk_emb = torch.stack([b["spk_emb"] for b in batch]).unsqueeze(1)  # (B,1,192)

        return {
            "all_token": all_token,
            "text_len": text_len,
            "target_len": target_len,
            "loss_mask": loss_mask,
            "w2v_input_features": inputs["input_features"],
            "w2v_attention_mask": inputs["attention_mask"],
            "spk_emb": spk_emb,
        }
