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
import torchvision.transforms as tv
from transformers import CLIPVisionModelWithProjection, AutoImageProcessor
from PIL import Image

IMAGE_SIZE = 224
IMAGE_MEAN = 0.48145466, 0.4578275, 0.40821073
IMAGE_STD = 0.26862954, 0.26130258, 0.27577711

def load_vision_embedding_model(model_name):
    if model_name == "wav2clip":
        hf_repo = f"openai/clip-vit-base-patch16" # BASE : 149.62M
        model = CLIPVisionModelWithProjection.from_pretrained(hf_repo)
        processor = AutoImageProcessor.from_pretrained(hf_repo, use_fast=True)
    elif model_name == "audioclip":
        sys.path.append("/home/daeyong/gaudio_retrieval_evaluation/audioclip/AudioCLIP")
        from model import AudioCLIP
        model = AudioCLIP(pretrained="/home/daeyong/gaudio_retrieval_evaluation/audioclip/AudioCLIP/assets/AudioCLIP-Full-Training.pt") # AudioCLIP-Full-Training.pt 저장 경로
        processor = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Resize(IMAGE_SIZE, interpolation=Image.BICUBIC),
            tv.transforms.CenterCrop(IMAGE_SIZE),
            tv.transforms.Normalize(IMAGE_MEAN, IMAGE_STD)
        ])
    model.eval()
    return model, processor
