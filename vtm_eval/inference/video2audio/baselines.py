import os
import json
import torch
from time import time
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
from vtm_eval.extractors.vision_embedding import load_vision_embedding_model
from torchcodec.decoders import VideoDecoder
import numpy as np

def load_music_features(args):
    model_name = args.model_type
    embedding_dict = torch.load(f"{args.embedding_path}/{model_name}.pt", weights_only=False)
    music_ids = list(embedding_dict.keys())
    music_features = torch.cat([torch.from_numpy(embedding_dict[_id]) for _id in music_ids], dim=0)
    return music_ids, music_features

def retrieval_fn(video_feature, music_features, music_ids):
    video_feature = torch.nn.functional.normalize(video_feature, dim=-1)
    music_features = torch.nn.functional.normalize(music_features, dim=-1)
    similarity = torch.matmul(video_feature, music_features.T).squeeze(0)
    ranked_idx = similarity.argsort(dim=-1, descending=True).tolist()
    ranked_music = [music_ids[i] for i in ranked_idx]
    similarity_score = [similarity[i].item() for i in ranked_idx]
    return ranked_music, similarity_score

def main(args):

    music_ids, music_features = load_music_features(args)
    embedding_model, processor = load_vision_embedding_model(args.model_type)
    embedding_model.eval()
    embedding_model.to(args.device)

    real_time_factor_start = time()
    vr = VideoDecoder(args.video_path)
    num_frames = vr.metadata.num_frames
    average_fps = vr.metadata.average_fps
    duration_seconds = vr.metadata.duration_seconds
    num_seconds = int(np.floor(duration_seconds))
    frame_idx = (np.arange(num_seconds) * int(round(average_fps))).clip(0, num_frames-1)
    video_frames = vr.get_frames_at(indices=frame_idx).data  # T x C x H x W
    video_frames = processor(video_frames, return_tensors="pt")
    pixel_values = video_frames["pixel_values"]
    pixel_values = pixel_values.to(args.device)
    with torch.no_grad():
        video_features = embedding_model(pixel_values=pixel_values)
    video_embeddings = video_features.image_embeds.detach().cpu()
    video_feature = video_embeddings.mean(dim=0, keepdim=True)
    ranked_music, similarity_score = retrieval_fn(video_feature, music_features, music_ids)
    real_time_factor_end = time()
    real_time_factor = real_time_factor_end - real_time_factor_start
    retrieval_results = [{"music_id": mid, "score": score}for mid, score in zip(ranked_music[:args.top_k], similarity_score[:args.top_k])]
    result = {
        "input_video_path": args.video_path,
        "real_time_factor": real_time_factor,
        "retrieval_results": retrieval_results,
        "model_type": args.model_type,
    }
    save_name = args.video_path.split("/")[-1].replace(".mp4", "")
    model_name = args.model_type
    os.makedirs(f"exp/v2a_baseline/{save_name}", exist_ok=True)
    with open(f"exp/v2a_baseline/{save_name}/{model_name}.json", "w") as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="exp/v2a_baseline")
    parser.add_argument("--video_captioner", type=str, default="llava-hf/LLaVA-NeXT-Video-7B-hf")
    parser.add_argument("--model_type", type=str, default="wav2clip")
    parser.add_argument("--embedding_path", type=str, default="cache/audio_embedding/cc_bgm")
    parser.add_argument("--dataset_path", type=str, default="/data/dean/evalset/gaudio/data/clip")
    parser.add_argument("--file_name", type=str, default="StreetFoodFighter_s01e04/video/82bd1fe8-caa8-49e0-b80b-afc11bc8a2c5.mp4")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()
    args.video_path = os.path.join(args.dataset_path, args.file_name)
    main(args)
