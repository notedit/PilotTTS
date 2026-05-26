import torch
from transformers import Wav2Vec2BertModel


def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    if length_dim == 0:
        raise ValueError(f"length_dim cannot be 0: {length_dim}")
    bs = int(len(lengths))
    if maxlen is None:
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)
    else:
        assert xs is None
        assert maxlen >= int(max(lengths))
    seq_range = torch.arange(0, maxlen, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)
        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def build_semantic_model(stats_path, w2v_path="facebook/w2v-bert-2.0"):
    semantic_model = Wav2Vec2BertModel.from_pretrained(w2v_path)
    semantic_model.eval()
    stat_mean_var = torch.load(stats_path)
    semantic_mean = stat_mean_var["mean"]
    semantic_std = torch.sqrt(stat_mean_var["var"])
    return semantic_model, semantic_mean, semantic_std
