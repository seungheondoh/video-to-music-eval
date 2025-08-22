"""
Calculates the cosine similarity between video and audio embeddings,
ranks the audios for each video based on similarity, and saves the
results to a JSON file.
"""

import os
import json
import torch
import torch.nn.functional as F


def calculate_similarity_and_rank(video_feature, audio_features, audio_ids):
    """
    Calculates cosine similarity between a single video feature and multiple
    audio features, then ranks the audios by similarity.

    Args:
        video_feature (torch.Tensor): The feature tensor for a single video.
        audio_features (torch.Tensor): A batch of feature tensors for audios.
        audio_ids (list): A list of identifiers for the audio files.

    Returns:
        tuple: A tuple containing:
            - list: A list of audio IDs, ranked by descending similarity.
            - list: A list of corresponding similarity scores.
    """
    # Normalize features to unit vectors for cosine similarity calculation
    video_feature = F.normalize(video_feature, dim=-1)
    audio_features = F.normalize(audio_features, dim=-1)

    # Calculate similarity using matrix multiplication (dot product of normalized vectors)
    similarity = torch.matmul(video_feature, audio_features.transpose(0, 1)).squeeze(0)

    # Sort the similarity scores in descending order
    ranked_indices = similarity.argsort(descending=True).tolist()

    # Get the ranked audio IDs and their scores
    ranked_audio_ids = [audio_ids[i] for i in ranked_indices]
    similarity_scores = [similarity[i].item() for i in ranked_indices]

    return ranked_audio_ids, similarity_scores


def generate_similarity_json(video_embeddings_path, audio_embeddings_path, output_json_path):
    """
    Loads video and audio embeddings, computes their similarities, and saves
    the ranked results to a JSON file.

    Args:
        video_embeddings_path (str): Path to the video embeddings file (.pt).
        audio_embeddings_path (str): Path to the audio embeddings file (.pt).
        output_json_path (str): Path to save the output JSON file.
    """
    # 1. Load embeddings from files
    print(f"Loading video embeddings from {video_embeddings_path}...")
    video_embeddings = torch.load(video_embeddings_path, map_location="cpu")
    print(f"Loaded {len(video_embeddings)} video embeddings.")

    print(f"Loading audio embeddings from {audio_embeddings_path}...")
    audio_embeddings = torch.load(audio_embeddings_path, map_location="cpu")
    print(f"Loaded {len(audio_embeddings)} audio embeddings.")

    # Prepare audio features for batch processing
    audio_ids = list(audio_embeddings.keys())
    audio_features = torch.stack([audio_embeddings[aid].squeeze() for aid in audio_ids], dim=0)

    results = []

    # 2. Calculate similarity for each video against all audios
    print("Calculating similarities...")
    for video_path, video_feature in video_embeddings.items():
        ranked_audio_ids, scores = calculate_similarity_and_rank(
            video_feature.unsqueeze(0), audio_features, audio_ids
        )

        ranked_audios_with_scores = [
            {"audio_id": aid, "score": score} for aid, score in zip(ranked_audio_ids, scores)
        ]

        results.append({
            "video_path": video_path,
            "top_similar_audios": ranked_audios_with_scores
        })

    # 3. Save the results to a JSON file
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Successfully saved similarity results to {output_json_path}")


def main():
    """
    This block runs when the script is executed directly.
    It calculates and saves similarity results for different embedding models.
    """
    BASE_EMBEDDINGS_DIR = "/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval"

    # Qwen model embeddings (video caption)
    generate_similarity_json(
        video_embeddings_path=os.path.join(BASE_EMBEDDINGS_DIR, "video_embeddings", "qwen.pt"),
        audio_embeddings_path=os.path.join(BASE_EMBEDDINGS_DIR, "audio_embeddings", "qwen.pt"),
        output_json_path=os.path.join(BASE_EMBEDDINGS_DIR, "results", "qwen.json")
    )
    
    # Qwen model embeddings (music caption)
    generate_similarity_json(
        video_embeddings_path=os.path.join(BASE_EMBEDDINGS_DIR, "video_embeddings", "qwen_music.pt"),
        audio_embeddings_path=os.path.join(BASE_EMBEDDINGS_DIR, "audio_embeddings", "qwen.pt"),
        output_json_path=os.path.join(BASE_EMBEDDINGS_DIR, "results", "qwen_music.json")
    )

    # AudioClip model embeddings
    generate_similarity_json(
        video_embeddings_path=os.path.join(BASE_EMBEDDINGS_DIR, "video_embeddings", "audioclip.pt"),
        audio_embeddings_path=os.path.join(BASE_EMBEDDINGS_DIR, "audio_embeddings", "audioclip.pt"),
        output_json_path=os.path.join(BASE_EMBEDDINGS_DIR, "results", "audioclip.json")
    )

    # Wav2Clip model embeddings
    generate_similarity_json(
        video_embeddings_path=os.path.join(BASE_EMBEDDINGS_DIR, "video_embeddings", "wav2clip.pt"),
        audio_embeddings_path=os.path.join(BASE_EMBEDDINGS_DIR, "audio_embeddings", "wav2clip.pt"),
        output_json_path=os.path.join(BASE_EMBEDDINGS_DIR, "results", "wav2clip.json")
    )


if __name__ == "__main__":
    main()

