#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates text embeddings for video captions using a SentenceTransformer model.

This script reads a JSON file containing video metadata (paths and captions),
extracts the captions, and computes their embeddings using a specified
SentenceTransformer model. The resulting embeddings are stored in a dictionary
mapping video IDs (derived from filenames) to their corresponding embedding
tensors. This dictionary is then saved to a PyTorch file (.pt).
"""

import argparse
import json
import os
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def main(args):
    """
    Main function to generate and save text embeddings.

    Args:
        args: Command-line arguments from argparse.
    """
    # 1. Load video metadata from the input JSON file.
    print(f"Loading captions from: {args.json_path}")
    with open(args.json_path, "r", encoding="utf-8") as f:
        video_items = json.load(f)

    # 2. Prepare lists of video IDs and captions for processing.
    video_ids = []
    captions = []
    for item in video_items:
        video_path = item.get("video_path")
        caption = item.get("caption", "")

        # Skip items with no valid caption.
        if not caption or not isinstance(caption, str):
            continue

        # Use the filename (without extension) as the video ID.
        video_id = Path(video_path).stem if video_path else None
        if not video_id:
            continue

        video_ids.append(video_id)
        captions.append(caption)

    if not video_ids:
        raise RuntimeError("No valid captions found in the input JSON.")

    # 3. Load the pre-trained SentenceTransformer model.
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, device=args.device)

    # 4. Generate embeddings for all captions in batches.
    print("Generating embeddings...")
    embeddings = model.encode(
        captions,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,  # Normalize for cosine similarity.
    )

    # 5. Create a dictionary mapping video IDs to their embedding tensors.
    embedding_dict = {
        video_id: torch.from_numpy(emb)
        for video_id, emb in zip(video_ids, embeddings)
    }

    # 6. Save the dictionary to a .pt file.
    output_dir = os.path.dirname(args.out_path)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(embedding_dict, args.out_path)

    print(f"\n✅ Saved embeddings to: {args.out_path}")
    print(f"   - Entries: {len(embedding_dict)}")
    print(f"   - Model: {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text embeddings for video captions."
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/video_captions/video_to_music_captions_qwen.json",
        help="Path to the input JSON file with video captions.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/video_embeddings/qwen_music.pt",
        help="Path to save the output .pt file with embeddings.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for computation (e.g., 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for the embedding model.",
    )
    args = parser.parse_args()
    main(args)
