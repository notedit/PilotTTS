import torch


class BeamHypotheses:
    """Maintains n-best beam hypotheses sorted by length-normalized score."""

    def __init__(self, num_beams, length_penalty, early_stopping=False):
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        return len(self.beams)

    def add(self, hyp, sum_logprobs, generated_len=None):
        if generated_len is not None and generated_len > 0:
            score = sum_logprobs / (generated_len ** self.length_penalty)
        else:
            score = sum_logprobs

        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted(
                    [(s, idx) for idx, (s, _) in enumerate(self.beams)]
                )
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        if len(self) < self.num_beams:
            return False
        if self.early_stopping:
            return True
        highest_attainable = best_sum_logprobs / (cur_len ** self.length_penalty)
        return self.worst_score >= highest_attainable


def expand_cache(cache, num_beams):
    """Expand KV-cache from batch_size=1 to batch_size=num_beams.

    Creates a NEW DynamicCache to avoid corrupting model-internal references.
    """
    if hasattr(cache, "key_cache"):
        from transformers.cache_utils import DynamicCache
        new_cache = DynamicCache()
        new_cache.key_cache = [
            t.repeat_interleave(num_beams, dim=0) for t in cache.key_cache
        ]
        new_cache.value_cache = [
            t.repeat_interleave(num_beams, dim=0) for t in cache.value_cache
        ]
        new_cache._seen_tokens = cache._seen_tokens
        return new_cache

    return tuple(
        tuple(t.repeat_interleave(num_beams, dim=0) for t in layer)
        for layer in cache
    )


def reorder_cache(cache, beam_idx):
    """Reorder KV-cache to match selected beam indices.

    Creates new tensors to avoid in-place corruption.
    """
    if hasattr(cache, "key_cache"):
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i].index_select(0, beam_idx)
            cache.value_cache[i] = cache.value_cache[i].index_select(0, beam_idx)
        return cache

    return tuple(
        tuple(t.index_select(0, beam_idx.to(t.device)) for t in layer)
        for layer in cache
    )


def cleanup_cache(cache):
    """Explicitly release all tensors in a KV-cache to free GPU memory."""
    if cache is None:
        return
    if hasattr(cache, "key_cache"):
        cache.key_cache.clear()
        cache.value_cache.clear()
        cache._seen_tokens = 0
