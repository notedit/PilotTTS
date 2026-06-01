# PilotTTS: A Disciplined Modular Recipe for Competitive Speech Synthesis

<p align="center">
    <img src="assert/Introduction.png" width="600" />
</p>

<p align="center">
    <a href="README.md">English</a> &nbsp;|&nbsp; 中文
</p>

<p align="center">
    📑 <a href="https://arxiv.org/abs/2605.27258">论文</a> &nbsp;|&nbsp; 🤗 <a href="https://huggingface.co/AmapVoice/PilotTTS">HuggingFace</a> &nbsp;|&nbsp; 🤖 <a href="https://www.modelscope.cn/models/AmapVoice/PilotTTS">ModelScope</a> &nbsp;|&nbsp; 🎧 <a href="https://amapvoice.github.io/PilotTTS/">演示样例</a>
</p>


## 最新动态 📝

- **[即将发布]** 正在进行14+方言能力的扩展，模型权重即将发布
- **[2026.05]** 发布 Pilot-TTS 基础模型和指令模型权重



## 亮点 🔥
**PilotTTS** 是一个基于大语言模型（LLM）的文本到语音（TTS）系统。它采用全开源模型构建极简化架构，并在严格的数据工程实践下实现了具有竞争力的性能。
### 关键特性：
- **一个由全开源算子构建的数据处理流水线：**  我们设计了一个多阶段的、涵盖质量评估与增强、标注以及质量过滤能力的、且全部算子都建立在公开可用工具之上的数据处理流水线。该流水线将负责互联网音频转化为干净且富有标注信息的训练数据，在显著降低成本的同时实现高质量数据产出。
- **内容一致性与说话人相似度情感控制：** 我们的模型在 Seed-TTS 测试集上，获得了极高的说话人相似度（0.862）以及极具竞争力的内容准确性（CER 0.87%），达到业界领先水平。
- **情感和副语言控制：** 支持11 类情感（Happy、Sad、Fear、Angry、Contempt、Serious、Surprise、Blue、Concern、Disgust、Psychology）的控制合成以及4类副语言（LAUGH、BREATH、CRY、COUGH）控制合成。
- **方言合成：** 支持14种中文方言，并且支持跨方言合成，尤其擅长中文普通话到目标方言的合成。

## 安装 ⚙️

### 克隆仓库

```bash
git clone https://github.com/xxx/pilot-tts.git
cd pilot-tts
```

### 环境配置 

```bash
conda create -n pilot-tts python=3.10 -y
conda activate pilot-tts
pip install -r requirements.txt
```

### 模型下载

#### 1. Pilot-TTS 模型（我们的权重）

```python
# ModelScope
from modelscope import snapshot_download
snapshot_download('xxx/Pilot-TTS', local_dir='pretrained_models/')

# HuggingFace
from huggingface_hub import snapshot_download
snapshot_download('xxx/Pilot-TTS', local_dir='pretrained_models/')
```

包含：`pilot_tts.pt`、`pilot_tts_instruct.pt` 和 `tokenizer/`。

#### 2. 第三方开源模型

从各自的开源项目下载以下依赖：

```python
from modelscope import snapshot_download

# Qwen3-0.6B（LLM 骨干网络）
snapshot_download('Qwen/Qwen3-0.6B', local_dir='pretrained_models/Qwen3-0.6B')

# CosyVoice3（flow-matching 声码器，包含 campplus.onnx）
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/CosyVoice3-0.5B')
```

```python
from huggingface_hub import snapshot_download

# w2v-bert-2.0（音频特征提取器）
snapshot_download('facebook/w2v-bert-2.0', local_dir='pretrained_models/w2v-bert-2.0')
```

> 注：`wav2vec2bert_stats.pt`（来自 [MaskGCT](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct)）已包含在 Pilot-TTS 模型包中。

#### 最终目录结构

```
pretrained_models/
├── pilot_tts.pt              # 基础模型（零样本声音克隆）
├── pilot_tts_instruct.pt     # 指令模型（情感、副语言、方言）
├── Qwen3-0.6B/              # LLM 骨干网络（来自 Qwen）
├── w2v-bert-2.0/            # 音频特征提取器（来自 Meta）
├── wav2vec2bert_stats.pt    # 特征归一化统计（来自 MaskGCT）
└── CosyVoice3-0.5B/        # Flow-matching 声码器（来自 FunAudioLLM）
```

## 快速开始 📖

一键运行所有推理示例：

```bash
python demo.py
```

## 推理

### Python API

```python
from demo import load_engine, synthesize

# 零样本声音克隆（基础模型）
engine = load_engine(
    config_path="configs/infer_pilot_tts.yaml",
    checkpoint="pretrained_models/pilot_tts.pt",
)
synthesize(engine, text="你好，世界！",
           prompt_wav="assert/prompt.wav",
           output_path="output/clone.wav")

# 加载指令模型（情感、副语言、方言）
engine_instruct = load_engine(
    config_path="configs/infer_pilot_tts_instruct.yaml",
    checkpoint="pretrained_models/pilot_tts_instruct.pt",
)

# 情感合成
synthesize(engine_instruct, text="今天天气真好啊！",
           prompt_wav="assert/prompt.wav",
           emotion="happy", output_path="output/happy.wav")

# 副语言
synthesize(engine_instruct, text="这太好笑了<|LAUGH|>停不下来",
           prompt_wav="assert/prompt.wav",
           output_path="output/laugh.wav")

# 方言（河南话）
synthesize(engine_instruct, text="中不中啊，咱俩一块儿去吃胡辣汤吧",
           prompt_wav="assert/prompt.wav",
           language="zh-henan", output_path="output/henan.wav")
```

### 命令行

```bash
# 零样本声音克隆（基础模型）
python inference.py \
    --checkpoint pretrained_models/pilot_tts.pt \
    --prompt-wav assert/prompt.wav \
    --text "需要合成的目标文本" \
    --output output/zeroshot.wav

# 情感合成（指令模型）
python inference.py \
    --config configs/infer_pilot_tts_instruct.yaml \
    --checkpoint pretrained_models/pilot_tts_instruct.pt \
    --prompt-wav assert/prompt.wav \
    --text "今天天气真好啊，我们去公园玩吧！" \
    --emotion happy \
    --output output/emotion.wav

# 副语言（指令模型）
python inference.py \
    --config configs/infer_pilot_tts_instruct.yaml \
    --checkpoint pretrained_models/pilot_tts_instruct.pt \
    --prompt-wav assert/prompt.wav \
    --text "这个笑话太好笑了<|LAUGH|>我真的忍不住" \
    --output output/paralang.wav

# 方言合成（指令模型）
python inference.py \
    --config configs/infer_pilot_tts_instruct.yaml \
    --checkpoint pretrained_models/pilot_tts_instruct.pt \
    --prompt-wav assert/prompt.wav \
    --text "中不中啊，咱俩一块儿去吃胡辣汤吧" \
    --language zh-henan \
    --output output/dialect.wav
```

### 支持的控制功能

| 功能 | 用法 | 所需模型 |
|------|------|----------|
| 声音克隆 | 提供参考音频 | 两者均可 |
| 情感 | `--emotion <标签>` | 指令模型 |
| 副语言 | 在文本中插入标签 | 指令模型 |
| 方言 | `--language <方言>` | 指令模型 |

**情感标签：**

| 标签 | 情感 | 标签 | 情感 |
|------|------|------|------|
| `happy` | 开心 | `sad` | 悲伤 |
| `angry` | 愤怒 | `surprise` | 惊讶 |
| `fear` | 恐惧 | `disgust` | 厌恶 |
| `serious` | 严肃 | `concern` | 关切 |
| `blue` | 忧郁 | `disdain` | 轻蔑 |
| `neutral` | 中性/平静 | `psychology` | 心理活动 |
| `unknown` | 不指定情感 | | |

**副语言标签：**

| 标签 | 说明 |
|------|------|
| `<\|LAUGH\|>` | 笑声 |
| `<\|BREATH\|>` | 呼吸声 |
| `<\|COUGH\|>` | 咳嗽 |
| `<\|CRY\|>` | 哭泣声 |
| `<\|LAUGH_SPAN\|>...<\|/LAUGH_SPAN\|>` | 包裹笑声文本 |

**方言标签：**

| 标签 | 方言 | 标签 | 方言 |
|------|------|------|------|
| `zh-dongbei` | 东北话 | `zh-shandong` | 山东话 |
| `zh-henan` | 河南话 | `zh-shan1xi` | 山西话 |
| `zh-minnan` | 闽南语 |  `zh-gansu` | 甘肃话 |
| `zh-ningxia` | 宁夏话 | `zh-shanghai` | 上海话 |
| `zh-chongqing` | 重庆话 | `zh-hubei` | 湖北话 |
| `zh-hunan` | 湖南话 | `zh-jiangxi` | 江西话 |
| `zh-guizhou` | 贵州话 | `zh-yunnan` | 云南话 |

## WebUI

启动基于 Gradio 的交互式界面：

```bash
python webui.py --port 9000
```

## 项目结构

```
pilot-tts/
├── configs/                     # 推理配置（按 checkpoint 区分）
├── demo.py                      # 完整示例（全部推理模式）
├── inference.py                 # 命令行推理入口
├── webui.py                     # Gradio WebUI
├── asset/                       # 示例参考音频
├── pilot_voice/                 # 核心模型代码
│   ├── engine.py                # InferenceEngine 推理流水线
│   ├── model.py                 # AR 模型（Qwen3 骨干 + 音频 token）
│   ├── sampling.py              # RAS 采样（源自 VALL-E 2）
│   ├── utils.py                 # 工具函数
│   ├── modules/                 # Conformer + Perceiver 模块
│   └── tools/                   # 音频与文本处理工具
├── third_party/
│   ├── cosyvoice/               # Flow-matching 声码器
│   └── Matcha-TTS/              # Flow matching 依赖
├── tokenizer/                   # 含特殊 token 的自定义分词器
├── pretrained_models/           # 模型权重（不在 git 中）
└── requirements.txt
```

## 致谢

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) — Flow-matching 与声码器
- [Qwen3](https://github.com/QwenLM/Qwen3) — LLM 骨干网络
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) — Flow matching 框架
- [MaskGCT](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct) — wav2vec2bert 特征统计


## 引用

```bibtex
@article{pilottts2026,
      title={PilotTTS: A Disciplined Modular Recipe for Competitive Speech Synthesis},
      author={Bowen Li and Shaotong Guo and Zhen Wang and Yang Xiang and Mingli Jin and Yihang Lin and Jiahui Zhao and Weibo Xiong and Dongrui Li and Keming Chen and Yunze Gao and Yuze Zhou and Zeyang Lin and Yue Liu},
      year={2026},
      journal={arXiv preprint arXiv:2605.27258}
}
```

## 许可证

Apache-2.0

