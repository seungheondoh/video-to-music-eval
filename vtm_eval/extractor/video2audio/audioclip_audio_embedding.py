import sys
import os
import argparse
import torch
import librosa
import numpy as np
from tqdm import tqdm

def main(args):
    """
    Extracts and saves AudioCLIP embeddings from audio files.
    """
    # Add AudioCLIP module path
    if args.audioclip_module_path not in sys.path:
        sys.path.append(args.audioclip_module_path)
    from model import AudioCLIP

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    # --- Load Model ---
    print("Loading AudioCLIP model...")
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)
    audio_embedding_model = AudioCLIP(pretrained=args.model_path).to(device).eval()
    print(f"Model loaded on {device}.")

    # --- Extract Embeddings ---
    process_audio_files(args.audio_root, audio_embedding_model, device, args.output_path)

def process_audio_files(audio_root, model, device, output_path):
    """
    Processes audio files in the specified folder to extract and save embeddings.
    """
    try:
        if not os.path.isdir(audio_root):
            raise FileNotFoundError(f"Audio root directory not found at {audio_root}")

        # Get a list of audio files to process (supports .mp3, .wav, .flac)
        supported_extensions = (".mp3", ".wav", ".flac")
        audio_files = sorted([f for f in os.listdir(audio_root) if f.lower().endswith(supported_extensions)])
        
        if not audio_files:
            print(f"No supported audio files {supported_extensions} found in {audio_root}")
            return

        embedding_dict = {}
        for audio_file in tqdm(audio_files, desc="Extracting AudioCLIP embeddings"):
            audio_path = os.path.join(audio_root, audio_file)
            
            # Extract embedding
            embedding = get_audioclip_embedding(model, audio_path, device)
            audio_id = os.path.splitext(audio_file)[0]
            embedding_dict[audio_id] = embedding

        # Save results
        save_embeddings(embedding_dict, output_path)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def save_embeddings(embedding_dict, output_path):
    """
    Saves the extracted embeddings to a file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(embedding_dict, output_path)
    print(f"✅ Saved {len(embedding_dict)} embeddings to: {output_path}")

@torch.no_grad()
def get_audioclip_embedding(model, audio_path, device):
    """
    Extracts an AudioCLIP embedding from a single audio file.
    """
    try:
        track, _ = librosa.load(audio_path, sr=44100, dtype=np.float32)
        track_tensor = torch.from_numpy(track).unsqueeze(0).to(device)
        # Model inference
        ((audio_features, _, _), _), _ = model(audio=track_tensor)
        return audio_features.squeeze(0).cpu()
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract AudioCLIP embeddings from audio files.")
    
    parser.add_argument("--audio_root", type=str, required=True,
                        help="Path to the directory containing audio files.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the output embeddings file (.pt).")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pretrained AudioCLIP model file (.pt).")
    parser.add_argument("--audioclip_module_path", type=str, required=True,
                        help="Path to the AudioCLIP module directory.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to use for inference (cuda or cpu).")

    args = parser.parse_args()
    main(args)
