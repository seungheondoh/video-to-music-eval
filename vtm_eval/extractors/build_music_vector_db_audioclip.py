import sys
import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
sys.path.append("/home/daeyong/gaudio_retrieval_evaluation/audioclip/AudioCLIP")
from model import AudioCLIP

# AudioCLIP-Full-Training.pt 저장 경로
audio_embedding_model = AudioCLIP(pretrained="/home/daeyong/gaudio_retrieval_evaluation/audioclip/AudioCLIP/assets/AudioCLIP-Full-Training.pt").eval() 

@torch.no_grad()
def get_audioclip_embedding(model, audio_path):
    track, _ = librosa.load(audio_path, sr=44100, dtype=np.float32)
    track_tensor = torch.from_numpy(track).unsqueeze(0)
    ((audio_features, _, _), _), _ = model(audio=track_tensor)
    return audio_features.squeeze(0)  # (1024,)

# 경로 설정
audio_root = "/home/daeyong/gaudio_retrieval_evaluation/ossl/audio"
output_path = "/home/daeyong/gaudio_retrieval_evaluation/cache/audio_embedding/ossl/audioclip_full.pt"

# 폴더 순회 및 임베딩 추출
folder_list = sorted([
    name for name in os.listdir(audio_root)
    if os.path.isdir(os.path.join(audio_root, name))
])

embedding_dict = {}

for folder in tqdm(folder_list, desc="Extracting AudioCLIP embeddings"):
    audio_path = os.path.join(audio_root, folder, "music.wav")
    if not os.path.exists(audio_path):
        print(f"⚠️ Skipping (not found): {audio_path}")
        continue
    embedding = get_audioclip_embedding(audio_embedding_model, audio_path)
    embedding_dict[folder] = embedding

# 저장
os.makedirs(os.path.dirname(output_path), exist_ok=True)
torch.save(embedding_dict, output_path)
print(f"✅ Saved {len(embedding_dict)} embeddings to: {output_path}")
