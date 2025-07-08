import os
import json
import torch
import numpy as np
from time import time
from tqdm import tqdm
from glob import glob
from PIL import Image
from torchvision.transforms import ToPILImage
from torchcodec.decoders import VideoDecoder
from vtm_eval.extractors.vision_embedding import load_vision_embedding_model


def load_music_features(model_type, embedding_path):
    embedding_dict = torch.load(f"{embedding_path}/{model_type}.pt", weights_only=False)
    music_ids = list(embedding_dict.keys())
    # music_features = torch.cat([embedding_dict[_id] for _id in music_ids], dim=0)
    music_features = torch.stack([embedding_dict[_id] for _id in music_ids], dim=0)
    return music_ids, music_features


def retrieval_fn(video_feature, music_features, music_ids):
    video_feature = torch.nn.functional.normalize(video_feature, dim=-1)
    music_features = torch.nn.functional.normalize(music_features, dim=-1)
    similarity = torch.matmul(video_feature, music_features.T).squeeze(0)
    ranked_idx = similarity.argsort(dim=-1, descending=True).tolist()
    ranked_music = [music_ids[i] for i in ranked_idx]
    similarity_score = [similarity[i].item() for i in ranked_idx]
    return ranked_music, similarity_score


def video_to_music_retrieval(video_path, model_type, music_ids, music_features, embedding_model, processor, device, top_k):
    vr = VideoDecoder(video_path)
    num_frames = vr.metadata.num_frames
    average_fps = vr.metadata.average_fps
    duration_seconds = vr.metadata.duration_seconds
    num_seconds = int(np.floor(duration_seconds))
    frame_idx = (np.arange(num_seconds) * int(round(average_fps))).clip(0, num_frames - 1)
    video_frames = vr.get_frames_at(indices=frame_idx).data  # T x C x H x W

    if model_type == "audioclip":
        processed = []
        for frame in video_frames:
            pil_img = ToPILImage()(frame.cpu())
            tensor = processor(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                image_feature = embedding_model.encode_image(tensor)
                image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
                processed.append(image_feature)
        video_feature = torch.cat(processed, dim=0).mean(dim=0, keepdim=True).cpu()

    ranked_music, similarity_score = retrieval_fn(video_feature, music_features, music_ids)
    retrieval_results = [{"music_id": mid, "score": score} for mid, score in zip(ranked_music[:top_k], similarity_score[:top_k])]

    return {
        "video_path": video_path,
        "retrieval_results": retrieval_results
    }


def main(args):
    # Load music embeddings
    music_ids, music_features = load_music_features(args.model_type, args.embedding_path)

    # Load visual model
    embedding_model, processor = load_vision_embedding_model(args.model_type)
    embedding_model.to(args.device).eval()

    # Load previously saved results if exist
    output_path = "results/video_music_retrieval_results_audioclip.json"
    os.makedirs("results", exist_ok=True)
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
                model_type=args.model_type,
                music_ids=music_ids,
                music_features=music_features,
                embedding_model=embedding_model,
                processor=processor,
                device=args.device,
                top_k=args.top_k
            )
            output["real_time_factor"] = round(time() - start, 4)

            results.append(output)

            # Save after each video
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"❌ Failed on {video_path}: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="audioclip")
    parser.add_argument("--embedding_path", type=str, default="/home/daeyong/gaudio_retrieval_evaluation/cache/audio_embedding/ossl")
    parser.add_argument("--dataset_path", type=str, default="/home/daeyong/gaudio_retrieval_evaluation/ossl/video")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--top_k", type=int, default=200)
    args = parser.parse_args()

    main(args)
