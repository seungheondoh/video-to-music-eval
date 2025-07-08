import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
import wav2clip

@torch.no_grad()
def get_wav2clip_embedding(model, audio_path):
    track, _ = librosa.load(audio_path, sr=16000, dtype=np.float32)
    embedding = wav2clip.embed_audio(track, model) # track[:16000*30]
    return embedding

def main(args):
    # Load dataset
    df = pd.read_csv("ossl/metadata/ossl_metadata.csv")

    # Load model
    model = wav2clip.get_model().eval()

    save_dict = {}
    for i, row in tqdm(df.iterrows(), total=len(df), desc="🔍 Extracting wav2clip embeddings"):
        audio_id = row["id"]
        audio_path = os.path.join(args.audio_path, audio_id, "music.wav")

        if not os.path.exists(audio_path):
            print(f"⚠️ Missing: {audio_path}")
            continue

        embedding = get_wav2clip_embedding(model, audio_path)
        save_dict[audio_id] = torch.from_numpy(embedding)

    # Save
    save_path = f"cache/audio_embedding/{args.dataset_name}/wav2clip.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(save_dict, save_path)
    print(f"✅ Saved {len(save_dict)} embeddings to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ossl")
    parser.add_argument("--audio_path", type=str, default="/home/daeyong/gaudio_retrieval_evaluation/ossl/audio")
    args = parser.parse_args()

    main(args)
