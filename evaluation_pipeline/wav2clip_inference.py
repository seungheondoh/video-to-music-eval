import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
import wav2clip
import argparse

@torch.no_grad()
def get_wav2clip_embedding(model, audio_path: str) -> np.ndarray:
    """
    Loads an audio file, computes its embedding using the Wav2Clip model.

    Args:
        model: The loaded Wav2Clip model.
        audio_path (str): The path to the audio file.

    Returns:
        np.ndarray: The computed audio embedding.
    """
    # Load audio file at 16kHz sample rate as a 32-bit float array
    track, _ = librosa.load(audio_path, sr=16000, dtype=np.float32)
    
    # Generate the embedding for the audio track
    embedding = wav2clip.embed_audio(track, model)
    return embedding

def main(args):
    """
    Main function to extract embeddings for all audio files in a directory
    and save them into a single PyTorch tensor file.
    """

    # Load the pre-trained Wav2Clip model and set it to evaluation mode
    print("Loading Wav2Clip model...")
    model = wav2clip.get_model().eval()
    print("Model loaded successfully.")

    # Get a sorted list of all .mp3 files in the audio directory
    try:
        wav_files = sorted([f for f in os.listdir(args.audio_root) if f.lower().endswith(".mp3")])
        if not wav_files:
            print(f"⚠️ No .mp3 files found in {args.audio_root}")
            return
    except FileNotFoundError:
        print(f"❌ Error: The directory {args.audio_root} was not found.")
        return

    embeddings_dict = {}
    # Iterate over each audio file and compute its embedding
    for wav_file in tqdm(wav_files, desc="🔍 Extracting Wav2Clip embeddings"):
        audio_id = os.path.splitext(wav_file)[0]  # Use filename without extension as ID
        audio_path = os.path.join(args.audio_root, wav_file)

        if not os.path.exists(audio_path):
            print(f"⚠️ Skipping missing file: {audio_path}")
            continue

        try:
            embedding = get_wav2clip_embedding(model, audio_path)
            # Store the embedding as a PyTorch tensor in the dictionary
            embeddings_dict[audio_id] = torch.from_numpy(embedding)
        except Exception as e:
            print(f"❌ Error processing {audio_path}: {e}")
            continue

    if not embeddings_dict:
        print("No embeddings were generated.")
        return

    # Create the directory for the save path if it doesn't exist
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # Save the dictionary of embeddings to a file
    torch.save(embeddings_dict, args.save_path)
    print(f"✅ Saved {len(embeddings_dict)} embeddings to: {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Wav2Clip embeddings from audio files.")
    parser.add_argument('--audio_root', type=str, required=True, help='The root directory containing the audio files.')
    parser.add_argument('--save_path', type=str, required=True, help='The path to save the output embeddings file (e.g., embeddings.pt).')
    args = parser.parse_args()
    main(args)
