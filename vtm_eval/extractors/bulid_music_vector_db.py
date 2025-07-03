from vtm_eval.extractors.text_embedding import load_text_embedding_model, get_qwen_embedding, get_bert_embedding
from vtm_eval.extractors.audio_embedding import load_audio_embedding_model, get_audioclip_embedding, get_wav2clip_embedding
from vtm_eval.extractors.vtmr_embedding import load_vtmr_embedding_model, get_vtmr_audio_embedding
from datasets import load_dataset
import argparse
import os
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="segmented-cc-audio")
parser.add_argument("--audio_path", type=str, default="/data/dean/evalset")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Embedding-0.6B") # bert-base-uncased
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

def main():
    if args.dataset_name == "ossl":
        folder_name = "ossl"
    dataset = load_dataset(f"seungheondoh/{args.dataset_name}", split="train")

    if args.model_name in ["wav2clip", "audioclip"]:
        model = load_audio_embedding_model(args.model_name)
        embedding_type = "audio"
    else:
        model, tokenizer = load_text_embedding_model(args.model_name)
        embedding_type = "text"
    model.to(args.device)
    save_dict = {}
    for item in tqdm(dataset):
        elif args.dataset_name == "ossl":
            text = item["caption"]
            audio_path = None # TODO: add audio embedding path in here
        if args.model_name == "bert-base-uncased":
            embedding = get_bert_embedding(model, tokenizer, text)
        elif args.model_name == "Qwen/Qwen3-Embedding-0.6B":
            embedding = get_qwen_embedding(model, tokenizer, text)
        elif args.model_name == "wav2clip":
            embedding = get_wav2clip_embedding(model, audio_path)
        elif args.model_name == "audioclip":
            embedding = get_audioclip_embedding(model, audio_path)
        elif args.model_name == "vtmr":
            embedding = get_vtmr_audio_embedding(model, npy_path)
        save_dict[item["id"]] = embedding
    model_name = args.model_name.split("/")[-1]
    save_path = f"cache/{embedding_type}_embedding/{folder_name}/{model_name}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(save_dict, save_path)

if __name__ == "__main__":
    main()
