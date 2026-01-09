import os
import numpy as np
import torch
from tqdm import tqdm
import laion_clap
import argparse

def get_clap_embeddings_from_dir(
    model: laion_clap.CLAP_Module, 
    input_dir: str, 
) -> dict:
    """
    Generates CLAP audio embeddings for all .mp3 or .wav files in a specified directory.
    
    Args:
        model (laion_clap.CLAP_Module): The loaded CLAP model object.
        input_dir (str): The directory path where the audio files are located.

    Returns:
        dict: A dictionary with 'audio_id' as key and NumPy embedding array as value.
    """
    embeddings_dict = {}
    
    # 1. Check if the input directory exists
    if not os.path.isdir(input_dir):
        print(f"❌ Error: Input directory '{input_dir}' does not exist.")
        return embeddings_dict

    # 2. Get a list of all .mp3 or .wav files in the directory.
    audio_files = [f for f in os.listdir(input_dir) if f.endswith('.mp3') or f.endswith('.wav')]
    if not audio_files:
        print(f"⚠️ Warning: No .mp3 or .wav files found in '{input_dir}'.")
        return embeddings_dict

    print(f"Processing a total of {len(audio_files)} audio files.")

    # 3. Iterate through each file and generate embeddings.
    for filename in tqdm(sorted(audio_files), desc="Generating CLAP embeddings"):
        # Create the full file path.
        full_path = os.path.join(input_dir, filename)
        
        # Extract 'audio_id' by removing the extension from the filename.
        audio_id = os.path.splitext(filename)[0]
        
        try:
            # Use the get_audio_embedding_from_filelist method of the laion_clap model.
            audio_embed = model.get_audio_embedding_from_filelist(
                x=[full_path], 
                use_tensor=False
            )
            
            # ❗ Key change: Store the NumPy array directly without calling .tolist().
            embeddings_dict[audio_id] = audio_embed.squeeze()
        
        except Exception as e:
            print(f"\n❌ Error: An error occurred while processing '{filename}': {e}")
            continue
            
    return embeddings_dict

def save_embeddings_to_npz(data: dict, output_path: str):
    """
    Saves the embedding dictionary to a compressed .npz file.
    """
    try:
        # Save and compress the dictionary's key-value pairs directly into the file.
        np.savez_compressed(output_path, **data)
        print(f"\n✅ Embeddings successfully saved to '{output_path}'.")
    except Exception as e:
        print(f"❌ Error: An error occurred while saving the file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save CLAP audio embeddings from a directory.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing audio files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output .npz file.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'.")

    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        print("Loading CLAP model...")
        print(f"Using device: {device}")
        
        model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
        model.load_ckpt()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"❌ Fatal error during model loading: {e}")
        exit()

    # 2. Generate embeddings
    embeddings = get_clap_embeddings_from_dir(model, args.input_dir)

    # 3. Save embeddings
    if embeddings:
        save_embeddings_to_npz(embeddings, args.output_path)