import os
import pandas as pd
import torch
import json
from time import time
from tqdm import tqdm
import random
import subprocess
import tempfile
import shutil
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
from vtm_eval.extractors.text_embedding import load_text_embedding_model, get_qwen_embedding, get_bert_embedding

def get_random_clip(video_path, temp_dir):
    # Get video duration in seconds
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        duration = float(result.stdout.strip())
    except:
        return video_path  # fallback to original if any error

    if duration <= 60:
        return video_path  # use whole video

    # Pick a random start time
    start_time = random.uniform(0, duration - 60)

    # Make temp output path
    basename = os.path.basename(video_path)
    temp_output_path = os.path.join(temp_dir, f"clip_{basename}")

    # Run ffmpeg to extract the clip
    cmd = [
        "ffmpeg", "-y", "-ss", str(start_time), "-t", "60",
        "-i", video_path, "-c", "copy", temp_output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return temp_output_path if os.path.exists(temp_output_path) else video_path


def get_video_caption(caption_model, caption_processor, video_path):
    conversation = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is happening in the video?"}, {"type": "video", "path": video_path},],
        },
    ]
    inputs = caption_processor.apply_chat_template(conversation, num_frames=8, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    model_device = next(caption_model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    # inputs = {k: v.to(args.device) for k,v in inputs.items()}
    with torch.no_grad():
        out = caption_model.generate(**inputs, max_new_tokens=1024)
    caption = caption_processor.batch_decode(out.detach().cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    caption = caption[0].split("ASSISTANT: ")[-1]
    return caption

def load_music_features(text_model, embedding_path):
    model_name = text_model.split("/")[-1]
    embedding_dict = torch.load(f"{embedding_path}/{model_name}.pt")
    music_ids = list(embedding_dict.keys())
    music_features = torch.cat([embedding_dict[_id] for _id in music_ids], dim=0)
    return music_ids, music_features

def retrieval_fn(video_feature, music_features, music_ids):
    video_feature = torch.nn.functional.normalize(video_feature, dim=-1)
    music_features = torch.nn.functional.normalize(music_features, dim=-1)
    similarity = torch.matmul(video_feature, music_features.T).squeeze(0)
    ranked_idx = similarity.argsort(dim=-1, descending=True).tolist()
    ranked_music = [music_ids[i] for i in ranked_idx]
    similarity_score = [similarity[i].item() for i in ranked_idx]
    return ranked_music, similarity_score

def video_to_music_retrieval(
    video_path: str,
    text_model: str,
    music_ids: list,
    music_features: torch.Tensor,
    caption_model,
    caption_processor,
    embedding_model,
    tokenizer,
    top_k: int = 200,
):

    real_time_factor_start = time()
    video_caption = get_video_caption(caption_model, caption_processor, video_path)
    if text_model == "bert-base-uncased":
        video_feature = get_bert_embedding(embedding_model, tokenizer, video_caption)
    elif text_model == "Qwen/Qwen3-Embedding-0.6B":
        instruction = "Retrieve music caption that matches the following video caption"
        video_feature = get_qwen_embedding(embedding_model, tokenizer, video_caption, instruction)
    else:
        raise ValueError(f"Unsupported text model: {text_model}")
    ranked_music, similarity_score = retrieval_fn(video_feature, music_features, music_ids)
    real_time_factor_end = time()
    real_time_factor = real_time_factor_end - real_time_factor_start
    retrieval_results = [{"music_id": mid, "score": score} for mid, score in zip(ranked_music[:top_k], similarity_score[:top_k])]
    result = {
        "video_path": video_path,
        "video_caption": video_caption,
        "real_time_factor": real_time_factor,
        "retrieval_results": retrieval_results,
    }
    return result

def main(args):
    # temp dir for clips
    temp_dir = tempfile.mkdtemp()
    
    # main()에서 1번만 로드
    music_ids, music_features = load_music_features(args.text_model, args.embedding_path)
    
    # Load models
    embedding_model, tokenizer = load_text_embedding_model(args.text_model)
    embedding_model.eval().to("cuda:0")

    caption_model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        args.video_captioner,
        torch_dtype=torch.float16,
        device_map="auto", 
        max_memory={0: "10GiB", 1: "9GiB"}
    ).eval()

    caption_processor = LlavaNextVideoProcessor.from_pretrained(args.video_captioner)

    # Iterate over all .mp4 files
    results = []
    for fname in tqdm(sorted(os.listdir(args.dataset_path))):
        if not fname.endswith(".mp4"):
            continue

        video_path = os.path.join(args.dataset_path, fname)
        # if video_path in already_done:
        #     print(f"⏩ Skipping already processed: {video_path}")
        #     continue
        
        # 🔸 랜덤 60초 클립 생성
        processed_video_path = get_random_clip(video_path, temp_dir)
        
        print(f"▶ Processing: {video_path}")
        try:
            start_time = time()
            output = video_to_music_retrieval(
                video_path=processed_video_path, # 랜덤 클립 경로
                text_model=args.text_model,
                music_ids=music_ids,
                music_features=music_features,
                caption_model=caption_model,
                caption_processor=caption_processor,
                embedding_model=embedding_model,
                tokenizer=tokenizer,
                top_k=args.top_k
            )
            elapsed = time() - start_time

            results.append({
                "video_path": output["video_path"],
                "video_caption": output["video_caption"],
                "real_time_factor": elapsed,
                "retrieval_results": output["retrieval_results"]
            })
            
            # ✅ 결과를 실시간 저장
            with open("results/video_music_retrieval_results_bert.json", "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ Failed to process {fname}: {e}")
        
    # 🔸 Cleanup
    shutil.rmtree(temp_dir)

    # 결과를 json으로 저장
    with open("results/video_music_retrieval_results_bert.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_captioner", type=str, default="llava-hf/LLaVA-NeXT-Video-7B-hf")
    parser.add_argument("--text_model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding_path", type=str, default="/home/daeyong/gaudio_retrieval_evaluation/cache/text_embedding/ossl")
    parser.add_argument("--dataset_path", type=str, default="/home/daeyong/gaudio_retrieval_evaluation/ossl/video")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--top_k", type=int, default=200)
    args = parser.parse_args()
    main(args)
    
    
# python baselines_daeyong.py \ 
#   --video_captioner llava-hf/LLaVA-NeXT-Video-7B-hf \
#   --text_model Qwen/Qwen3-Embedding-0.6B \
#   --embedding_path /home/daeyong/gaudio_retrieval_evaluation/cache/text_embedding/ossl \
#   --dataset_path /home/daeyong/gaudio_retrieval_evaluation/ossl/video \
#   --device cuda:0 \
#   --top_k 200
