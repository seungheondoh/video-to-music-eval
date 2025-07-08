import os
import json
import torch
import numpy as np
from time import time
from tqdm import tqdm
from glob import glob
from torch.nn.functional import normalize
from transformers import CLIPVisionModelWithProjection, AutoImageProcessor
from torchcodec.decoders import VideoDecoder

def load_music_features(embedding_path):
    embedding_dict = torch.load(f"{embedding_path}/wav2clip.pt", weights_only=False)
    music_ids = list(embedding_dict.keys())

    fixed_embeddings = []
    for _id in music_ids:
        emb = embedding_dict[_id]
        if emb.dim() == 2 and emb.shape[0] == 1:
            emb = emb.squeeze(0)
        elif emb.dim() == 2 and emb.shape[1] == 1:
            emb = emb.squeeze(1)
        elif emb.dim() != 1:
            raise ValueError(f"Invalid shape for embedding {_id}: {emb.shape}")
        fixed_embeddings.append(emb)

    music_features = torch.stack(fixed_embeddings, dim=0)  # (N, 512)
    return music_ids, music_features


def load_vision_embedding_model():
    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch16")
    processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    return model.eval(), processor


def get_video_embedding(video_path, model, processor, device):
    vr = VideoDecoder(video_path)
    num_frames = vr.metadata.num_frames
    average_fps = vr.metadata.average_fps
    duration_seconds = vr.metadata.duration_seconds
    num_seconds = int(np.floor(duration_seconds))
    frame_idx = (np.arange(num_seconds) * int(round(average_fps))).clip(0, num_frames - 1)
    video_frames = vr.get_frames_at(indices=frame_idx).data  # T x C x H x W

    processed = processor(video_frames, return_tensors="pt")
    pixel_values = processed["pixel_values"].to(device)

    with torch.no_grad():
        features = model(pixel_values=pixel_values)
        video_feature = features.image_embeds.mean(dim=0, keepdim=True).cpu()  # (1, 512)

    return video_feature


def retrieval_fn(video_feature, music_features, music_ids):
    video_feature = normalize(video_feature, dim=-1)
    music_features = normalize(music_features, dim=-1)
    similarity = torch.matmul(video_feature, music_features.T).squeeze(0)
    ranked_idx = similarity.argsort(dim=-1, descending=True).tolist()
    ranked_music = [music_ids[i] for i in ranked_idx]
    similarity_score = [similarity[i].item() for i in ranked_idx]
    return ranked_music, similarity_score


def video_to_music_retrieval(video_path, embedding_model, processor, device, music_ids, music_features, top_k):
    video_feature = get_video_embedding(video_path, embedding_model, processor, device)
    ranked_music, similarity_score = retrieval_fn(video_feature, music_features, music_ids)
    return {
        "video_path": video_path,
        "retrieval_results": [
            {"music_id": mid, "score": score}
            for mid, score in zip(ranked_music[:top_k], similarity_score[:top_k])
        ]
    }


def main(args):
    music_ids, music_features = load_music_features(args.embedding_path)
    embedding_model, processor = load_vision_embedding_model()
    embedding_model.to(args.device).eval()

    output_path = "/home/daeyong/gaudio_retrieval_evaluation/video-to-music-eval/vtm_eval/inference/video2audio/results/video_music_retrieval_results_wav2clip.json"

    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            results = json.load(f)
        already_processed = set([r["video_path"] for r in results])
    else:
        results = []
        already_processed = set()

    video_files = sorted(glob(os.path.join(args.dataset_path, "*.mp4")))

    for video_path in tqdm(video_files, desc="🔍 Processing videos"):
        if video_path in already_processed:
            print(f"⏩ Skipping already processed: {video_path}")
            continue

        try:
            print(f"▶ Processing: {video_path}")
            start = time()
            output = video_to_music_retrieval(
                video_path=video_path,
                embedding_model=embedding_model,
                processor=processor,
                device=args.device,
                music_ids=music_ids,
                music_features=music_features,
                top_k=args.top_k
            )
            output["real_time_factor"] = round(time() - start, 4)
            results.append(output)

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"❌ Failed on {video_path}: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_path", type=str, default="/home/daeyong/gaudio_retrieval_evaluation/cache/audio_embedding/ossl")
    parser.add_argument("--dataset_path", type=str, default="/home/daeyong/gaudio_retrieval_evaluation/ossl/video")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--top_k", type=int, default=200)
    args = parser.parse_args()

    main(args)
