import os
import json
import torch
from time import time
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
from vtm_eval.extractors.text_embedding import load_text_embedding_model, get_qwen_embedding, get_bert_embedding

def get_video_caption(caption_model, caption_processor, video_path):
    conversation = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is happening in the video?"}, {"type": "video", "path": video_path},],
        },
    ]
    inputs = caption_processor.apply_chat_template(conversation, num_frames=8, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    inputs = {k: v.to(args.device) for k,v in inputs.items()}
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
    embedding_path: str,
    device: str = "cuda",
    top_k: int = 200,
):
    music_ids, music_features = load_music_features(
        text_model=text_model,
        embedding_path=embedding_path
    )
    embedding_model, tokenizer = load_text_embedding_model(text_model)
    embedding_model.eval()
    embedding_model.to(device)
    caption_model = LlavaNextVideoForConditionalGeneration.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", torch_dtype=torch.float16)
    caption_processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
    caption_model.eval()
    caption_model.to(device)

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
    result = video_to_music_retrieval(
        video_path=args.video_path,
        text_model=args.text_model,
        embedding_path=args.embedding_path,
        device=args.device,
        top_k=args.top_k
    )
    print(result)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--video_path", type=str, default="/data/dean/evalset/gaudio/data/clip/StreetFoodFighter_s01e04/video/82bd1fe8-caa8-49e0-b80b-afc11bc8a2c5.mp4")
    parser.add_argument("--video_captioner", type=str, default="llava-hf/LLaVA-NeXT-Video-7B-hf") # Done
    parser.add_argument("--text_model", type=str, default="Qwen/Qwen3-Embedding-0.6B") # Done
    parser.add_argument("--embedding_path", type=str, default="/home/daeyong/gaudio_retrieval_evaluation/cache/text_embedding/ossl") # Done
    parser.add_argument("--dataset_path", type=str, default="/home/daeyong/gaudio_retrieval_evaluation/ossl/video")
    parser.add_argument("--file_name", type=str, default="video_id.mp4")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--top_k", type=int, default=200)
    args = parser.parse_args()
    args.video_path = os.path.join(args.dataset_path, args.file_name)
    main(args)
