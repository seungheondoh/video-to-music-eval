import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

def load_embeddings_from_npz(file_path: str) -> dict:
    """
    Loads a dictionary mapping 'audio_id' to embeddings from a .npz file.
    """
    try:
        print(f"Loading embeddings from '{file_path}'...")
        npzfile = np.load(file_path)
        embeddings_dict = {key: npzfile[key] for key in npzfile.keys()}
        print(f"✅ Loaded a total of {len(embeddings_dict)} embeddings.")
        return embeddings_dict
    except FileNotFoundError:
        print(f"❌ Error: Embedding file '{file_path}' not found.")
        return {}
    except Exception as e:
        print(f"❌ Error: An error occurred while loading the .npz file: {e}")
        return {}

def get_similarity_from_ids(
    gt_embeddings: dict,
    predicted_embeddings: dict,
    gt_audio_id: str, 
    predicted_audio_id: str
) -> float:
    """
    Finds GT and Predicted embeddings from different embedding dictionaries and calculates their similarity.
    """
    # Find GT ID in the gt_embeddings dictionary.
    gt_embedding_np = gt_embeddings.get(gt_audio_id)

    # Find Predicted ID in the predicted_embeddings dictionary.
    predicted_embedding_np = predicted_embeddings.get(predicted_audio_id)

    if gt_embedding_np is None:
        print(f"⚠️ Warning: GT embedding '{gt_audio_id}' not found.")
        return 0.0
    if predicted_embedding_np is None:
        print(f"⚠️ Warning: Predicted embedding '{predicted_audio_id}' not found.")
        return 0.0

    gt_embedding = torch.from_numpy(gt_embedding_np)
    predicted_embedding = torch.from_numpy(predicted_embedding_np)

    gt_embedding_normalized = F.normalize(gt_embedding, p=2, dim=-1)
    predicted_embedding_normalized = F.normalize(predicted_embedding, p=2, dim=-1)

    similarity_score = torch.dot(gt_embedding_normalized, predicted_embedding_normalized)

    return similarity_score.item()

def calculate_average_topk_similarity(
    json_path: str, 
    gt_embeddings: dict,
    predicted_embeddings: dict, 
    top_k: int
) -> float:
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: JSON file '{json_path}' not found.")
        return 0.0
    
    total_video_average_scores = []

    print(f"\n--- Starting Top-{top_k} Average Similarity Calculation ---")
    for item in data:
        video_path = item.get("video_path")
        top_similar_audios = item.get("top_similar_audios", [])
        
        if not video_path or not top_similar_audios:
            continue

        # Extract video_id (without extension) from video_path
        # e.g., /path/to/-11YMVRcYM4.mp4 -> -11YMVRcYM4
        gt_audio_id = os.path.splitext(os.path.basename(video_path))[0]
        
        # List to store similarity scores with Top-K audios for the current video
        scores_for_current_video = []
        
        # Iterate only up to Top-K
        for audio_info in top_similar_audios[:top_k]:
            predicted_audio_id = audio_info.get("audio_id")
            if not predicted_audio_id:
                continue
            
            # Calculate similarity based on IDs
            score = get_similarity_from_ids(
                gt_embeddings, # Pass GT embedding dictionary
                predicted_embeddings, # Pass Predicted embedding dictionary
                gt_audio_id, 
                predicted_audio_id
            )
            scores_for_current_video.append(score)
        
        # Calculate the average Top-K similarity for the current video
        if scores_for_current_video:
            average_score_for_video = sum(scores_for_current_video) / len(scores_for_current_video)
            total_video_average_scores.append(average_score_for_video)
            print(f"🎥 Video '{gt_audio_id}': Top-{top_k} average similarity = {average_score_for_video:.4f}")

    # Calculate the final average similarity for all videos
    if not total_video_average_scores:
        print("⚠️ Warning: Could not calculate average similarity because no valid data was found.")
        return 0.0
        
    final_average_similarity = sum(total_video_average_scores) / len(total_video_average_scores)

    return final_average_similarity, total_video_average_scores


def calculate_random_baseline_similarity(
    json_path: str,
    gt_embeddings: dict,
    predicted_embeddings: dict,
    num_samples: int
) -> float:
    """
    For each video in the JSON, randomly samples N audios, calculates the average similarity,
    and then computes the total average similarity across all videos to measure random baseline performance.
    (Version with separate GT and Predicted embedding sources)
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: JSON file '{json_path}' not found.")
        return 0.0

    # ❗ Change 1: The random sampling pool is taken from predicted_embeddings.
    all_predicted_ids = list(predicted_embeddings.keys())
    total_video_average_scores = []

    print(f"\n--- Starting Random Baseline (Top-{num_samples}) Average Similarity Calculation ---")
    for item in tqdm(data, desc="Calculating Random Baseline"):
        video_path = item.get("video_path")
        if not video_path:
            continue

        gt_audio_id = os.path.splitext(os.path.basename(video_path))[0]

        # ❗ Change 2: Check for the existence of gt_audio_id in gt_embeddings.
        if gt_audio_id not in gt_embeddings:
            print(f"\n⚠️ Warning: Ground Truth ID '{gt_audio_id}' not found in GT embedding pool, skipping.")
            continue

        # Randomly sample num_samples IDs from the entire Predicted audio pool
        random_predicted_ids = random.sample(all_predicted_ids, num_samples)

        scores_for_current_video = []
        for predicted_id in random_predicted_ids:
            # ❗ Change 3: Call the modified get_similarity_from_ids function.
            score = get_similarity_from_ids(
                gt_embeddings, 
                predicted_embeddings, 
                gt_audio_id, 
                predicted_id
            )
            scores_for_current_video.append(score)

        if scores_for_current_video:
            average_score_for_video = sum(scores_for_current_video) / len(scores_for_current_video)
            total_video_average_scores.append(average_score_for_video)

    if not total_video_average_scores:
        print("⚠️ Warning: Could not calculate average similarity because no valid data was found.")
        return 0.0

    final_average_similarity = sum(total_video_average_scores) / len(total_video_average_scores)
    return final_average_similarity

# --- Main execution block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate soft metrics for audio retrieval evaluation.")
    parser.add_argument("--gt_embeddings_path", type=str, required=True, help="Path to the ground truth embeddings .npz file.")
    parser.add_argument("--predicted_embeddings_path", type=str, required=True, help="Path to the predicted embeddings .npz file.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the JSON top k result file to be analyzed.")
    
    args = parser.parse_args()

    # 1. Load all audio embeddings into memory (only once)
    gt_embeddings = load_embeddings_from_npz(args.gt_embeddings_path)
    predicted_embeddings = load_embeddings_from_npz(args.predicted_embeddings_path)
    
    # 2. Set K values for averaging
    result = {}
    randoms = []
    for TOP_K in [10, 20, 50]:
        print(f"\n=== Calculating Top-{TOP_K} Similarity ===")

        # 3. Perform calculation only if embeddings were loaded successfully
        if gt_embeddings and predicted_embeddings:
            # 4. Call the function to calculate the final average similarity
            final_score, score_list = calculate_average_topk_similarity(
                json_path=args.json_path, 
                gt_embeddings=gt_embeddings,          # Pass GT embeddings
                predicted_embeddings=predicted_embeddings, # Pass Predicted embeddings
                top_k=TOP_K
            )    
            result[TOP_K] = round(final_score, 4)
        
    print(result)