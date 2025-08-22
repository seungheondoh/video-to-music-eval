#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates text embeddings from captions in a JSON file using a SentenceTransformer model.

This script reads a JSON file where each entry contains an 'audio_path' and a 'caption'.
It extracts the captions, generates embeddings using a specified model, and saves the
results as a dictionary mapping audio file stems to their corresponding embedding tensors
in a PyTorch (.pt) file.
"""

import os
import json
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def main(args):
    """
    Main function to generate and save text embeddings based on provided arguments.
    """
    # --- 1. Load data from JSON ---
    with open(args.json_path, "r", encoding="utf-8") as f:
        data_items = json.load(f)

    # --- 2. Prepare IDs and texts for embedding ---
    ids = []
    texts = []
    for item in data_items:
        audio_path = item.get("audio_path")
        caption = item.get("caption", "")

        # Skip items with no valid caption or audio path
        if not caption or not isinstance(caption, str) or not audio_path:
            continue

        # Use the filename (without extension) as the unique ID
        file_id = Path(audio_path).stem
        ids.append(file_id)
        texts.append(caption)

    if not ids:
        raise RuntimeError("No valid captions found in the input JSON.")

    # --- 3. Load the embedding model ---
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    model = SentenceTransformer(model_name, device=args.device)

    # --- 4. Generate embeddings in batches ---
    print(f"Generating embeddings for {len(texts)} texts using '{model_name}'...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # --- 5. Create a dictionary mapping IDs to embeddings ---
    embedding_dict = {
        file_id: torch.from_numpy(emb) for file_id, emb in zip(ids, embeddings)
    }

    # --- 6. Save the embeddings to a .pt file ---
    output_dir = os.path.dirname(args.out_path)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(embedding_dict, args.out_path)

    print(f"\n✅ Saved embeddings to: {args.out_path}")
    print(f"   - Total entries: {len(embedding_dict)}")
    print(f"   - Model used: {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text embeddings from captions in a JSON file."
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="/home/daeyong/gaudio_retrieval_evaluation/cc_bgm/audio_captions/audio_captions_qwen.json",
        help="Path to the input JSON file containing audio paths and captions.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="/home/daeyong/gaudio_retrieval_evaluation/cc_bgm/embeddings/qwen.pt",
        help="Path to save the output .pt file with embeddings.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for embedding generation (e.g., 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for the embedding model.",
    )
    args = parser.parse_args()
    main(args)
