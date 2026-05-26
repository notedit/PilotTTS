import threading
import uuid

import torch
from pathlib import Path
from transformers import AutoTokenizer

from cosyvoice.cli.cosyvoice import AutoModel

from .tools.audio import load_audio_norm, SpkEmbeddingExtractor
from cosyvoice.utils.file_utils import load_wav
from .tools.text import format_text_input
from .model import PilotVoice


class InferenceEngine:

    def __init__(self, config, device):
        self.config = config
        self.device = device

        self._load_model()
        self._load_tokenizer()
        self._load_vocoder()

        self.spk_extractor = SpkEmbeddingExtractor(
            config["spk_embedding"]["campplus_path"]
        )

    def _load_model(self):
        cfg = self.config["model"]
        self.model = PilotVoice(
            audio_tokens=cfg["audio_tokens"],
            pretrain_path=cfg["pretrain_path"],
            w2v_path=cfg.get("w2v_path", "facebook/w2v-bert-2.0"),
            w2v_stats_path=cfg.get("w2v_stats_path", ""),
            use_conditioning=True,
            use_spk_emb=True,
        )

        self.model.lock = threading.Lock()

        ckpt_path = Path(self.config["checkpoint_path"]).expanduser().absolute()
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        if "module" in checkpoint:
            state_dict = checkpoint["module"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        state_dict = {
            k.removeprefix("module."): v for k, v in state_dict.items()
        }
        self.model.load_state_dict(state_dict, strict=True)

        self.model.ar.load_semantic_model()

        self.model.eval()
        self.model = self.model.to(self.device)

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["tokenizer"]["path"]
        )

    def _load_vocoder(self):
        self.cosyvoice = AutoModel(
            model_dir=self.config["vocoder"]["model_dir"]
        )

    def ar_inference(self, text, spk_emb, prompt_audio, language=None):
        model = self.model.module if hasattr(self.model, "module") else self.model
        vocab_size = self.config["inference"]["vocab_size"]

        text_inp = format_text_input(
            "", text,
            include_prompt=False,
            language=language or self.config.get("language", "zh"),
        )
        text_tokens = self.tokenizer.encode(text_inp, add_special_tokens=False)
        text_tensor = torch.tensor(text_tokens, device=self.device).unsqueeze(0)
        tlen = torch.tensor([len(text_tokens)], device=self.device)

        proms_p = (
            (torch.zeros((text_tensor.shape[0], 1, 1)) + vocab_size)
            .long()
            .to(self.device)
        )
        proms_p_len = torch.tensor([1], device=self.device)

        ar_prompt = torch.concat(
            [text_tensor + vocab_size + 2, proms_p[:, :, 0]], dim=-1,
        )

        inf = self.config["inference"]
        with torch.no_grad():
            codes, codes_length = model.inference_ar(
                ar_prompt, text_tensor, tlen, proms_p, proms_p_len,
                top_p=inf["top_p"],
                top_k=inf["top_k"],
                sampling_temperature=inf["temperature"],
                spk_emb=spk_emb,
                prompt_audio=prompt_audio,
                num_beams=inf.get("num_beams", 1),
                length_penalty=inf.get("length_penalty", 0.0),
            )
        return codes, codes_length

    def vocoder_decode(self, codes, prompt_wav, text):
        model_input = self.cosyvoice.frontend.frontend_zero_shot(
            text, "", prompt_wav, self.cosyvoice.sample_rate, "",
        )
        this_uuid = str(uuid.uuid1())
        with self.cosyvoice.model.lock:
            self.cosyvoice.model.tts_speech_token_dict[this_uuid] = []
            self.cosyvoice.model.llm_end_dict[this_uuid] = False
            self.cosyvoice.model.hift_cache_dict[this_uuid] = None
        
        tts_speech = self.cosyvoice.model.token2wav(
            token=codes.squeeze(-1).unsqueeze(0),
            prompt_token=model_input["flow_prompt_speech_token"],
            prompt_feat=model_input["prompt_speech_feat"],
            embedding=model_input["flow_embedding"],
            token_offset=0,
            uuid=this_uuid,
            finalize=True,
        )

        with self.cosyvoice.model.lock:
            self.cosyvoice.model.tts_speech_token_dict.pop(this_uuid, None)
            self.cosyvoice.model.llm_end_dict.pop(this_uuid, None)
            self.cosyvoice.model.hift_cache_dict.pop(this_uuid, None)

        return tts_speech

    def synthesize(self, prompt_wav, text, language=None):
        raw_wav = load_wav(prompt_wav, 16000)

        norm_wav = load_audio_norm(prompt_wav)
        spk_emb = self.spk_extractor.extract(norm_wav, self.device)
        spk_emb = spk_emb.reshape((1, 1, -1))

        max_samples = 15 * 16000
        if raw_wav.shape[-1] > max_samples:
            raw_wav = raw_wav[..., :max_samples]
        prompt_audio = raw_wav

        codes, _ = self.ar_inference(text, spk_emb, prompt_audio, language=language)

        speech = self.vocoder_decode(codes, prompt_wav, text)
        return codes, speech
