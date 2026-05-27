import torch


def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25):
    
    sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(
        descending=True, stable=True
    )
    prob, indices = [], []
    cum_prob = 0.0
    for i in range(len(sorted_idx)):
        if cum_prob < top_p and len(prob) < top_k:
            cum_prob += sorted_value[i]
            prob.append(sorted_value[i])
            indices.append(sorted_idx[i])
        else:
            break
    prob = torch.tensor(prob).to(weighted_scores)
    indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
    return indices[prob.multinomial(1, replacement=True)]


def random_sampling(weighted_scores, decoded_tokens):
    return weighted_scores.softmax(dim=0).multinomial(1, replacement=True)


def ras_sampling(weighted_scores, decoded_tokens, top_p=0.95, top_k=25,
                 win_size=10, tau_r=0.1):
    top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
    rep_num = (
        torch.tensor(decoded_tokens[-win_size:]).to(weighted_scores.device) == top_ids
    ).sum().item()
    if rep_num >= win_size * tau_r:
        top_ids = random_sampling(weighted_scores, decoded_tokens)
    return top_ids
