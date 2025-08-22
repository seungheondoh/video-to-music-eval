import json
import os
from typing import List, Dict, Any
import numpy as np

def extract_id_from_path(file_path: str) -> str:
    """
    Extracts the file name without its extension from a given path.

    Args:
        file_path: The full path to the file.
                   e.g., "/path/to/ossl_1000_12.mp4"

    Returns:
        The base name of the file without the extension.
        e.g., "ossl_1000_12"
    """
    base_name = os.path.basename(file_path)
    return os.path.splitext(base_name)[0]

def calculate_recall_and_median_rank(results_json_path: str) -> Dict[str, float]:
    """
    Calculates recall@k and median rank from retrieval results.

    The function reads a JSON file where each entry contains a query video
    and a ranked list of retrieved audios. It checks if the correct audio
    (matching the video ID) is present in the top results.

    Args:
        results_json_path: Path to the JSON file with retrieval results.

    Returns:
        A dictionary containing recall@1, @5, @10, @50, and the median rank.
    """
    with open(results_json_path, 'r', encoding='utf-8') as f:
        results_data: List[Dict[str, Any]] = json.load(f)

    ranks = []
    for item in results_data:
        query_video_id = extract_id_from_path(item['video_path'])
        found_rank = None
        # Find the rank of the ground truth audio in the retrieved list.
        for idx, retrieved_audio in enumerate(item['top_similar_audios']):
            if retrieved_audio['audio_id'] == query_video_id:
                found_rank = idx + 1  # Ranks are 1-based.
                break
        
        # If the ground truth audio is not in the list, its rank is considered infinite.
        if found_rank is None:
            found_rank = float('inf')
        ranks.append(found_rank)

    ranks_array = np.array(ranks)

    # Calculate recall@k: percentage of queries where the correct item is in the top k.
    recall_at_1 = np.mean(ranks_array <= 1)
    recall_at_5 = np.mean(ranks_array <= 5)
    recall_at_10 = np.mean(ranks_array <= 10)
    recall_at_50 = np.mean(ranks_array <= 50)
    
    # Calculate median rank, ignoring queries where the item was not found.
    finite_ranks = ranks_array[ranks_array != float('inf')]
    median_rank = np.median(finite_ranks) if finite_ranks.size > 0 else float('inf')

    print(f"Recall@1:  {recall_at_1:.4f}")
    print(f"Recall@5:  {recall_at_5:.4f}")
    print(f"Recall@10: {recall_at_10:.4f}")
    print(f"Recall@50: {recall_at_50:.4f}")
    print(f"Median Rank: {median_rank}")

    return {
        "recall@1": recall_at_1,
        "recall@5": recall_at_5,
        "recall@10": recall_at_10,
        "recall@50": recall_at_50,
        "median_rank": median_rank
    }

def main():
    # Define paths to the result files for different models.
    base_path = "/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/results"
    result_files = {
        "Qwen": os.path.join(base_path, "qwen.json"),
        "QwenMusic": os.path.join(base_path, "qwen_music.json"),
        "AudioCLIP": os.path.join(base_path, "audioclip.json"),
        "Wav2CLIP": os.path.join(base_path, "wav2clip.json"),
    }

    # Calculate and print metrics for each model.
    print("=== Qwen Results (Video Caption) ===")
    metrics_qwen = calculate_recall_and_median_rank(result_files["Qwen"])
    
    print("=== Qwen Results (Music Caption) ===")
    metrics_qwen = calculate_recall_and_median_rank(result_files["QwenMusic"])

    print("\n=== AudioCLIP Results ===")
    metrics_audioclip = calculate_recall_and_median_rank(result_files["AudioCLIP"])

    print("\n=== Wav2CLIP Results ===")
    metrics_wav2clip = calculate_recall_and_median_rank(result_files["Wav2CLIP"])

if __name__ == "__main__":
    main()
