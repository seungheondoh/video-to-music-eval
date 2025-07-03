import os
from datasets import load_dataset, Dataset, DatasetDict
import ast
import torch
import argparse
from tqdm import tqdm
from torch import Tensor
import sys
import wav2clip
import librosa
import numpy as np

def load_audio_embedding_model(model_name):
    if model_name == "wav2clip":
        model = wav2clip.get_model()
    elif model_name == "audioclip":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.join(current_dir, "AudioCLIP"))
        from model import AudioCLIP
        model = AudioCLIP(pretrained="cache/AudioCLIP-Full-Training.pt")
    model.eval()
    return model

@torch.no_grad()
def get_wav2clip_embedding(model, audio_path):
    track, _ = librosa.load(audio_path, sr=16000, dtype=np.float32)
    embeddings = wav2clip.embed_audio(track, model)
    return embeddings

@torch.no_grad()
def get_audioclip_embedding(model, audio_path):
    track, _ = librosa.load(audio_path, sr=44100, dtype=np.float32)
    track_tensor = torch.from_numpy(track[:44100*30]).unsqueeze(0)
    ((audio_features, _, _), _), _ = model(audio=track_tensor)
    return audio_features
