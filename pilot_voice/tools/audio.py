import math

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import onnxruntime

from cosyvoice.utils.file_utils import load_wav


def load_audio_norm(audio_path, sr=16000):
    wav = load_wav(audio_path, sr)
    gain_db = -24 - math.log10(torch.mean(wav * wav).cpu().item() + 1e-16) * 10
    wav = torchaudio.functional.gain(wav, gain_db)
    wav = torch.clip(wav, min=-1, max=1)
    return wav


class SpkEmbeddingExtractor:

    def __init__(self, campplus_path):
        option = onnxruntime.SessionOptions()
        self.session = onnxruntime.InferenceSession(
            campplus_path,
            sess_options=option,
            providers=["CPUExecutionProvider"],
        )

    def extract(self, speech, device):
        feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.session.run(
            None,
            {self.session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()},
        )[0].flatten().tolist()
        return torch.tensor([embedding]).unsqueeze(0).to(device)
