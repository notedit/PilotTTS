# PilotTTS 训练复现(train/)

官方 repo 只放了推理代码。这套 `train/` 从 `pilot_voice/model.py` 和
`pilot_voice/engine.py` 反推出完整训练协议,产出的 checkpoint 与官方
`InferenceEngine`(`strict=True` 加载)**直接兼容**。

## 目录

```
train/
├── config.yaml            # 全部超参
├── prepare_data.py        # Step 2: FSQ speech token + CAM++ spk_emb 离线提取
├── compute_w2v_stats.py   # Step 3: w2v-bert layer-17 全局 mean/var
├── dataset.py             # 序列构造 / prompt 采样 / collate
├── train.py               # DDP 训练主脚本(分阶段)
├── sanity_check.py        # 正式训练前的对齐自检
└── README.md
```

## 协议速查(全部从源码反推,勿改)

**词表布局**(`audio_tokens=6563`,`vocab_size=6561`):

| id 区间 | 含义 |
|---|---|
| 0 – 6560 | CosyVoice3 FSQ speech codes(6561 个) |
| 6561 | BOA 分隔符(生成时被 `logits[..., 6561] = -inf` ban 掉) |
| 6562 | EOS |
| ≥ 6563 | 文本 token = Qwen tokenizer id + 6563 |

**训练序列**:`[text+6563] [6561] [codes] [6562]`,前面拼 33 个 conditioning
soft prefix(32 Perceiver latents + 1 个 zero-pad 到 1024 维的 CAM++ spk_emb)。

**loss**:只算 audio 段(codes + EOS)。这不是选择而是架构约束——`lm_head`
没有扩表(输出仍是 Qwen3 的 151936 维),偏移后的文本 id 超出输出维度。

**文本格式**:`<|0.00|><|zh|>你的文本<|0.02|>`,用 repo 自带 `tokenizer/`
(Qwen3 tokenizer + 时间戳/语种扩展 token),`add_special_tokens=False`。

---

## 复现步骤

### Step 0:环境与预训练组件

```bash
cd PilotTTS   # 以下命令全部在 repo 根目录执行
pip install -r requirements.txt

# 需要的预训练组件(与推理侧共用):
# pretrained_models/
# ├── Qwen3-0.6B/               # LLM 主干(训练必须有完整权重)
# ├── w2v-bert-2.0/             # facebook/w2v-bert-2.0, conditioning 特征
# └── CosyVoice3-0.5B/          # speech_tokenizer_v3.onnx + campplus.onnx + vocoder
```

### Step 1:准备原始数据 jsonl

每行一条,`speaker` 字段必须准确——训练时 conditioning prompt 靠它采样:

```json
{"utt": "spk001_0001", "wav": "/data/wav/spk001_0001.wav", "text": "今天天气不错。", "speaker": "spk001"}
```

**尽量保证每个说话人 ≥ 2 条 utterance**(原因见下面的坑位 #1)。
按说话人切 train/val(val 用 unseen speaker 才能真实反映 zero-shot 能力)。

### Step 2:离线提取 speech token + spk_emb

```bash
python train/prepare_data.py \
  --input data/raw_train.jsonl --output data/train.codes.jsonl \
  --speech_tokenizer pretrained_models/CosyVoice3-0.5B/speech_tokenizer_v3.onnx \
  --campplus pretrained_models/CosyVoice3-0.5B/campplus.onnx \
  --spk_emb_dir data/spk_emb --device cuda
# val 同理 -> data/val.codes.jsonl
```

提取逻辑镜像 cosyvoice frontend(whisper 128-mel → onnx)。**speech tokenizer
必须和 vocoder(token2wav)是同一套**,否则合成端 token 对不上。

### Step 3:计算 w2v-bert 统计量(必做)

```bash
python train/compute_w2v_stats.py \
  --manifest data/train.codes.jsonl \
  --w2v_path pretrained_models/w2v-bert-2.0 \
  --output pretrained_models/wav2vec2bert_stats.pt \
  --max_utts 30000
```

官方没放统计脚本但 `w2v_stats_path` 是硬依赖。必须在**你自己的数据分布**上算;
如果你手上有官方 `wav2vec2bert_stats.pt`(HF 权重包里带),直接用官方的更好——
和官方 ckpt 微调时**必须**用官方的。

### Step 4:sanity check

```bash
python train/sanity_check.py --config train/config.yaml
```

确认:文本段 id ≥ 6563、BOA=6561、EOS=6562、初始 loss ≈ 11.9(= ln 151936,
CE 在完整 lm_head 输出维度上算, lm_head 是 xavier 重新初始化的)、
perceiver/conformer 梯度非零。

### Step 5:Stage 1 —— 冻结 LLM,对齐 conditioner

`config.yaml` 保持 `stage: 1, lr: 1e-4`:

```bash
torchrun --nproc_per_node=8 train/train.py --config train/config.yaml
```

只训 conformer + perceiver + embed_tokens + lm_head(~0.2B 可训练)。
audio embedding 和 lm_head 都是随机初始化的,直接全量训会把 Qwen3 主干打乱。
跑到 loss 平台期即可(经验值 1-2 万 step / 数百小时数据规模)。

### Step 6:Stage 2 —— 全量微调

改 `config.yaml`:`stage: 2`,`lr: 2e-5`,`resume: exp/pilot_repro_stage1/step_XXXX.pt`,
`save_dir: exp/pilot_repro_stage2`,显存紧张打开 `gradient_checkpointing: true`:

```bash
torchrun --nproc_per_node=8 train/train.py --config train/config.yaml
```

### Step 7:推理验证

改 `configs/infer.yaml` 的 `checkpoint_path` 指向 stage2 ckpt,然后:

```bash
python inference.py   # 或 webui.py
```

ckpt 保存时已过滤 `ar.semantic_model.*`,`strict=True` 加载不会报错。
评测按官方口径跑 Seed-TTS test set 的 CER / speaker similarity。

---

## 坑位清单(按踩坑代价排序)

1. **prompt 内容泄漏(最致命)**。w2v-bert layer-17 是语义特征。如果训练时
   conditioning prompt 用目标音频本身,conditioner 会直接把内容"抄"给 LLM,
   训练 loss 漂亮,推理全崩(推理时 prompt 是参考音色,不含目标内容)。
   `dataset.py` 强制采样同说话人**另一条** utterance;单条说话人退化为随机
   裁剪,这类数据越少越好。
2. **w2v stats 不匹配**。换数据分布不重算 stats → conformer 输入 scale 漂移,
   前期 loss 曲线异常。微调官方 ckpt 时必须用官方 stats。
3. **checkpoint 污染**。任何把 w2v 前端挂到模型上的写法(`load_semantic_model`)
   都会让 `ar.semantic_model.*` 进 state_dict,engine 的 strict 加载直接炸。
   train.py 已把 w2v 前端放在模型外。
4. **mask 长度**。Perceiver 的 KV 里拼了 32 个 latents,attention mask 必须
   前补 32 个 True(`cond_mask_pad`)——模型代码已处理,自己改结构时注意。
5. **position_ids 偏移**。conds 占 position 0..32;推理侧已处理,如果你要接
   vLLM/自定义 serving,prefix 注入后所有 token 的 position 都要 +33。
6. **dtype**。bf16 autocast 下 fp32 的 conformer 输出进 bf16 LLM 没问题,
   但如果你手动转 half 权重,记得 `_compute_conds` 的 dtype cast 路径。

## 规模建议

官方达到 SOTA similarity 用的是大规模数据工程(数万小时级)。合理的复现
路径:先在 1k-5k 小时中文数据上验证 pipeline 收敛(CER 能到个位数),再
上量。数据量 < 500h 时建议 stage2 只放开 LLM 的上半部分 layers 或直接上
LoRA,否则 0.6B 主干会过拟合。jsonl manifest 在万小时级会成为瓶颈,届时
换 arrow/lance 分片 + 离线预提 w2v 特征(存 fp16,~100MB/h)把 GPU 利用率
拉满——w2v 在线提取大约占一张卡 15-20% 的算力。
