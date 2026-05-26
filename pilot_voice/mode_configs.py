MODE_CONFIGS = {
    "with_spkemb": {
        "use_conditioning": True,
        "use_spk_emb": True,
        "use_prompt_codecs": False,
        "include_prompt_in_text": False,
    },
    "no_spkemb": {
        "use_conditioning": True,
        "use_spk_emb": False,
        "use_prompt_codecs": False,
        "include_prompt_in_text": False,
    },
    "no_spkemb_conds": {
        "use_conditioning": False,
        "use_spk_emb": False,
        "use_prompt_codecs": True,
        "include_prompt_in_text": True,
    },
}
