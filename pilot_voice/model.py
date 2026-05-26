from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, SeamlessM4TFeatureExtractor

from .utils import make_pad_mask, build_semantic_model
from .sampling import ras_sampling
from .beam_search import BeamHypotheses, expand_cache, reorder_cache, cleanup_cache
from .modules.conformer_encoder import ConformerEncoder
from .modules.perceiver import PerceiverResampler


class AR(nn.Module):
    """Autoregressive speech token decoder built on a pretrained LLM backbone.

    Three operating modes controlled by two flags:
      - use_conditioning=True,  use_spk_emb=True  → wav2vec + conformer + perceiver + spk embedding prefix
      - use_conditioning=True,  use_spk_emb=False → wav2vec + conformer + perceiver prefix only
      - use_conditioning=False, use_spk_emb=False → pure token sequence, no conditioning prefix
    """

    def __init__(
        self,
        audio_tokens: int = 6563,
        pretrain_path: str = "Qwen/Qwen3-0.6B",
        w2v_path: str = "facebook/w2v-bert-2.0",
        w2v_stats_path: str = "",
        use_conditioning: bool = True,
        use_spk_emb: bool = True,
    ):
        super().__init__()
        self.audio_tokens = audio_tokens
        self.pretrain_path = pretrain_path
        self.use_conditioning = use_conditioning
        self.use_spk_emb = use_spk_emb

        self.ar_decoder = AutoModelForCausalLM.from_pretrained(pretrain_path)
        embed_dim = self.ar_decoder.model.embed_tokens.embedding_dim

        audio_embedding = nn.Embedding(
            audio_tokens, embed_dim,
            padding_idx=self.ar_decoder.model.embed_tokens.padding_idx,
        )
        nn.init.xavier_uniform_(audio_embedding.weight)

        with torch.no_grad():
            new_weight = torch.cat([
                audio_embedding.weight,
                self.ar_decoder.model.embed_tokens.weight,
            ])
        self.ar_decoder.model.embed_tokens.weight = nn.Parameter(new_weight, requires_grad=True)
        self.ar_decoder.model.embed_tokens.num_embeddings = new_weight.size(0)
        nn.init.xavier_uniform_(self.ar_decoder.lm_head.weight)

        # if use_conditioning:
        self.w2v_path = w2v_path
        self.w2v_stats_path = w2v_stats_path
        self.cond_num = 32

        self.conditioning_encoder = ConformerEncoder(
            input_size=1024, output_size=512, linear_units=2048,
            attention_heads=8, num_blocks=6, input_layer="linear",
        )
        self.perceiver_encoder = PerceiverResampler(
            embed_dim, dim_context=512, ff_mult=2, heads=8,
            num_latents=self.cond_num,
        )
        self.cond_mask_pad = nn.ConstantPad1d((self.cond_num, 0), True)

        for p in self.conditioning_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.perceiver_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def freeze_encoder(self):
        for param in self.conditioning_encoder.parameters():
            param.requires_grad = False
        for param in self.perceiver_encoder.parameters():
            param.requires_grad = False

    def freeze_llm(self):
        for param in self.ar_decoder.parameters():
            param.requires_grad = False

    def load_semantic_model(self, device="cpu"):
        assert self.use_conditioning
        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(self.w2v_path)
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            self.w2v_stats_path, w2v_path=self.w2v_path,
        )
        self.semantic_model.to(device)
        self.semantic_mean.to(device)
        self.semantic_std.to(device)
        for param in self.semantic_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _get_wav2vec_features(self, input_features, attention_mask):
        output = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = output.hidden_states[17]
        return (feat - self.semantic_mean.to(feat.device)) / self.semantic_std.to(feat.device)

    def _compute_conds(self, spk_cond_emb, attention_mask, spk_emb=None, dtype=None):
        """Run conditioning pipeline: conformer → perceiver → optional spk_emb prepend."""
        if dtype is not None:
            spk_cond_emb = spk_cond_emb.to(dtype)

        speech_cond, mask = self.conditioning_encoder(spk_cond_emb, attention_mask.sum(1))
        conds_mask = self.cond_mask_pad(mask.squeeze(1))
        conds = self.perceiver_encoder(speech_cond, conds_mask)

        if self.use_spk_emb and spk_emb is not None:
            embed_dim = self.ar_decoder.model.embed_tokens.embedding_dim
            spk_emb = F.pad(spk_emb, (0, embed_dim - spk_emb.shape[-1])).type_as(conds)
            conds = torch.cat((spk_emb, conds), dim=1)

        return conds

    def forward(
        self, all_token, text, text_len, target, target_len, proms_len,
        spk_emb=None, prompt_audio=None,
    ):
        """Training forward pass.

        Args:
            prompt_audio: dict with pre-extracted {"spk_cond_emb", "attention_mask"}, or None.
        """
        all_token_emb = self.ar_decoder.model.embed_tokens(all_token)
        T = all_token_emb.size(1)

        conds = None
        if self.use_conditioning and prompt_audio is not None:
            conds = self._compute_conds(
                prompt_audio["spk_cond_emb"],
                prompt_audio["attention_mask"].to(all_token.device),
                spk_emb=spk_emb,
                dtype=self.ar_decoder.dtype,
            )

        if conds is not None:
            all_token_emb = torch.cat((conds, all_token_emb), dim=1)
            input_masks = ~make_pad_mask(
                text_len + target_len + conds.size(1), maxlen=T + conds.size(1),
            )
        else:
            input_masks = ~make_pad_mask(text_len + target_len, maxlen=T)

        target_ar = all_token[:, 1:].to(torch.int32)
        out_all = self.ar_decoder(inputs_embeds=all_token_emb, attention_mask=input_masks)

        offset = conds.size(1) if conds is not None else 0
        out_ar = out_all.logits[:, offset:-1, :].permute(0, 2, 1)
        return out_ar, target_ar

    @torch.inference_mode()
    def generate(
        self, prompt_tokens, prompt_lengths, text_lengths,
        top_k=30, temperature=1.0, top_p=0.8, max_gen_len=4096,
        spk_emb=None, prompt_audio=None,
    ):
        """Autoregressive token generation with KV-cache.

        Args:
            prompt_audio: raw audio waveform list for feature extraction, or None.
        """
        bsz = len(prompt_tokens)
        device = prompt_tokens.device
        min_prompt_len = prompt_lengths.min().item()
        max_prompt_len = prompt_lengths.max().item()
        total_len = min(2048, max_gen_len + max_prompt_len)

        tokens = torch.full((bsz, total_len), 0, dtype=torch.long, device=device)
        tokens[:, :prompt_tokens.shape[1]] = prompt_tokens

        eos_reached = torch.tensor([False] * bsz, device=device)
        token_length = torch.tensor([min_prompt_len] * bsz, device=device)
        input_text_mask = ~make_pad_mask(prompt_lengths, maxlen=total_len)

        out_tokens = []
        cache = None
        prev_pos = 0

        conds = None
        if self.use_conditioning and prompt_audio is not None:
            inputs = self.extract_features(prompt_audio, sampling_rate=16000, return_tensors="pt")
            input_features = inputs["input_features"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            spk_cond_emb = self._get_wav2vec_features(input_features, attention_mask)
            conds = self._compute_conds(spk_cond_emb, attention_mask, spk_emb=spk_emb)

        for cur_pos in range(min_prompt_len, total_len):
            all_token = tokens[:, prev_pos:cur_pos]
            all_token_emb = self.ar_decoder.model.embed_tokens(all_token)
            T = all_token_emb.size(1)

            if conds is None:
                position_ids = torch.tensor(
                    [list(range(prev_pos, cur_pos))] * bsz, device=device,
                )
                input_masks = ~make_pad_mask(
                    torch.tensor([T] * bsz, device=device), maxlen=T,
                )
            else:
                cond_len = conds.size(1)
                position_ids = torch.tensor(
                    [list(range(prev_pos + cond_len, cur_pos + cond_len))] * bsz,
                    device=device,
                )
                input_masks = ~make_pad_mask(
                    torch.tensor([T] * bsz, device=device), maxlen=T,
                )
                if prev_pos == 0:
                    all_token_emb = torch.cat((conds, all_token_emb), dim=1)
                    position_ids = torch.tensor(
                        [list(range(0, cur_pos + cond_len))] * bsz, device=device,
                    )
                    input_masks = ~make_pad_mask(
                        torch.tensor([T + cond_len] * bsz, device=device),
                        maxlen=T + cond_len,
                    )

            out = self.ar_decoder(
                inputs_embeds=all_token_emb,
                attention_mask=input_masks,
                past_key_values=cache,
                position_ids=position_ids,
            )
            logits = out.logits
            cache = out.past_key_values

            logits = logits[:, :, :self.audio_tokens]
            logits[:, -1, self.audio_tokens - 2] = float("-inf")

            if temperature > 0:
                probs = torch.log_softmax(logits[:, -1] / temperature, dim=-1)
                next_token = ras_sampling(
                    probs.squeeze(0), out_tokens, top_p=top_p, top_k=top_k,
                ).unsqueeze(0)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token,
            )
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (next_token == self.audio_tokens - 1)
            token_length += 1 - eos_reached.long()
            prev_pos = cur_pos

            if all(eos_reached):
                break
            out_tokens.append(next_token)

        return tokens, token_length

    @torch.inference_mode()
    def generate_beam_search(
        self, prompt_tokens, prompt_lengths, text_lengths,
        num_beams=3, length_penalty=0.0, max_gen_len=4096,
        spk_emb=None, prompt_audio=None,
    ):
        """Autoregressive token generation using beam search with KV-cache.

        Returns the same (tokens, token_length) format as generate().
        """
        device = prompt_tokens.device
        min_prompt_len = prompt_lengths.min().item()
        max_prompt_len = prompt_lengths.max().item()
        total_len = min(2048, max_gen_len + max_prompt_len)
        eos_token = self.audio_tokens - 1
        vocab_size = self.audio_tokens

        tokens = torch.full((1, total_len), 0, dtype=torch.long, device=device)
        tokens[0, :prompt_tokens.shape[1]] = prompt_tokens[0]

        conds = None
        cond_len = 0
        if self.use_conditioning and prompt_audio is not None:
            inputs = self.extract_features(
                prompt_audio, sampling_rate=16000, return_tensors="pt",
            )
            input_features = inputs["input_features"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            spk_cond_emb = self._get_wav2vec_features(input_features, attention_mask)
            conds = self._compute_conds(
                spk_cond_emb, attention_mask, spk_emb=spk_emb,
            )
            cond_len = conds.size(1)

        # --- Phase 1: Prefill (process entire prompt with single beam) ---
        prompt_emb = self.ar_decoder.model.embed_tokens(tokens[:, :min_prompt_len])

        if conds is not None:
            prompt_emb = torch.cat((conds, prompt_emb), dim=1)
            position_ids = torch.arange(
                0, min_prompt_len + cond_len, device=device,
            ).unsqueeze(0)
            input_masks = torch.ones(
                1, min_prompt_len + cond_len, dtype=torch.bool, device=device,
            )
        else:
            position_ids = torch.arange(
                0, min_prompt_len, device=device,
            ).unsqueeze(0)
            input_masks = torch.ones(
                1, min_prompt_len, dtype=torch.bool, device=device,
            )

        from transformers.cache_utils import DynamicCache
        fresh_cache = DynamicCache()
        out = self.ar_decoder(
            inputs_embeds=prompt_emb,
            attention_mask=input_masks,
            past_key_values=fresh_cache,
            position_ids=position_ids,
        )
        cache = out.past_key_values
        prefill_logits = out.logits[:, -1, :vocab_size]
        del out

        # --- Phase 2: Expand to num_beams ---
        tokens = tokens.repeat(num_beams, 1)
        cache = expand_cache(cache, num_beams)

        beam_scores = torch.zeros(num_beams, dtype=torch.float, device=device)
        beam_scores[1:] = -1e9
        beam_hyps = BeamHypotheses(num_beams, length_penalty)

        next_logits = prefill_logits.expand(num_beams, -1).clone()
        last_pos = min_prompt_len - 1

        # --- Phase 3: Beam search decode ---
        for cur_pos in range(min_prompt_len, total_len):
            next_logits[:, self.audio_tokens - 2] = float("-inf")
            next_scores = torch.log_softmax(next_logits, dim=-1)
            next_scores = next_scores + beam_scores[:, None]

            next_scores_flat = next_scores.view(-1)
            n_candidates = 2 * num_beams
            top_scores, top_indices = torch.topk(
                next_scores_flat, n_candidates,
            )
            b_indices = top_indices // vocab_size
            t_indices = top_indices % vocab_size

            next_beam_scores = torch.full(
                (num_beams,), -1e9, device=device,
            )
            next_beam_tokens = torch.zeros(
                num_beams, dtype=torch.long, device=device,
            )
            next_beam_idx = torch.zeros(
                num_beams, dtype=torch.long, device=device,
            )
            beam_count = 0

            for rank in range(len(top_scores)):
                b = b_indices[rank]
                t = t_indices[rank]
                s = top_scores[rank]

                if t.item() == eos_token:
                    if rank >= num_beams:
                        continue
                    gen_len = cur_pos - min_prompt_len
                    if gen_len > 0:
                        gen = tokens[b, min_prompt_len:cur_pos].clone()
                        beam_hyps.add(gen, s.item(), generated_len=gen_len)
                else:
                    next_beam_scores[beam_count] = s
                    next_beam_tokens[beam_count] = t
                    next_beam_idx[beam_count] = b
                    beam_count += 1

                if beam_count == num_beams:
                    break

            if beam_count == 0:
                break

            if beam_count < num_beams:
                next_beam_tokens[beam_count:] = next_beam_tokens[0]
                next_beam_idx[beam_count:] = next_beam_idx[0]

            beam_scores = next_beam_scores
            tokens = tokens[next_beam_idx]
            tokens[:, cur_pos] = next_beam_tokens
            cache = reorder_cache(cache, next_beam_idx)
            last_pos = cur_pos

            gen_len = cur_pos - min_prompt_len + 1
            if beam_hyps.is_done(beam_scores.max().item(), gen_len):
                break

            token_emb = self.ar_decoder.model.embed_tokens(
                tokens[:, cur_pos:cur_pos + 1],
            )
            position_ids = torch.full(
                (num_beams, 1), cur_pos + cond_len,
                dtype=torch.long, device=device,
            )
            input_masks = torch.ones(
                num_beams, 1, dtype=torch.bool, device=device,
            )
            out = self.ar_decoder(
                inputs_embeds=token_emb,
                attention_mask=input_masks,
                past_key_values=cache,
                position_ids=position_ids,
            )
            cache = out.past_key_values
            next_logits = out.logits[:, -1, :vocab_size]
            del out

        # --- Phase 4: Finalize ---
        cleanup_cache(cache)
        del cache
        torch.cuda.empty_cache()

        for i in range(num_beams):
            gen_end = last_pos + 1
            gen = tokens[i, min_prompt_len:gen_end].clone()
            while len(gen) > 0 and gen[-1].item() == eos_token:
                gen = gen[:-1]
            gen_len = len(gen)
            if gen_len > 0:
                beam_hyps.add(gen, beam_scores[i].item(), generated_len=gen_len)

        if len(beam_hyps) == 0:
            result = torch.full((1, total_len), 0, dtype=torch.long, device=device)
            result[0, :min_prompt_len] = prompt_tokens[0, :min_prompt_len]
            return result, torch.tensor([min_prompt_len], device=device)

        best_score, best_tokens = max(beam_hyps.beams, key=lambda x: x[0])
        result = torch.full((1, total_len), 0, dtype=torch.long, device=device)
        result[0, :min_prompt_len] = prompt_tokens[0, :min_prompt_len]
        gen_len = len(best_tokens)
        result[0, min_prompt_len:min_prompt_len + gen_len] = best_tokens
        total_length = torch.tensor(
            [min_prompt_len + gen_len], device=device,
        )
        return result, total_length


class PilotVoice(nn.Module):

    def __init__(
        self,
        audio_tokens: int = 6563,
        pretrain_path: str = "Qwen/Qwen3-0.6B",
        w2v_path: str = "facebook/w2v-bert-2.0",
        w2v_stats_path: str = "",
        use_conditioning: bool = True,
        use_spk_emb: bool = True,
    ):
        super().__init__()
        self.ar = AR(
            audio_tokens=audio_tokens,
            pretrain_path=pretrain_path,
            w2v_path=w2v_path,
            w2v_stats_path=w2v_stats_path,
            use_conditioning=use_conditioning,
            use_spk_emb=use_spk_emb,
        )


    def forward(
        self, text, text_len, target, target_len, proms_len,
        all_token_list, spk_emb, prompt_audio, context=nullcontext,
    ):
        with context():
            return self.ar(
                all_token_list, text, text_len, target, target_len,
                proms_len, spk_emb, prompt_audio,
            )

    def inference_ar(
        self, ar_prompt, text, text_len, proms_p=None, proms_p_len=0,
        top_k=30, top_p=1.0, sampling_temperature=1.0,
        spk_emb=None, prompt_audio=None,
        num_beams=1, length_penalty=0.0,
    ):
        if num_beams > 1:
            target_ar, total_length = self.ar.generate_beam_search(
                ar_prompt, text_len + proms_p_len, text_len,
                num_beams=num_beams, length_penalty=length_penalty,
                spk_emb=spk_emb, prompt_audio=prompt_audio,
            )
        else:
            target_ar, total_length = self.ar.generate(
                ar_prompt, text_len + proms_p_len, text_len,
                top_k=top_k, top_p=top_p, temperature=sampling_temperature,
                spk_emb=spk_emb, prompt_audio=prompt_audio,
            )
        bsz = target_ar.shape[0]
        ar_list = []
        for i in range(bsz):
            ar_list.append(target_ar[i][text_len[i] + proms_p_len[i]:total_length[i]])
        return torch.cat(ar_list), total_length
