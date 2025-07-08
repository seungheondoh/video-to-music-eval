import os
import torch
import argparse
from tqdm import tqdm
import pandas as pd

from vtm_eval.extractors.text_embedding import (
    load_text_embedding_model,
    get_qwen_embedding,
    get_bert_embedding,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Embedding-0.6B")  # or "bert-base-uncased"
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

def main():
    df = pd.read_csv("ossl/metadata/ossl_metadata.csv")

    # 2. Load Text Embedding Model
    model, tokenizer = load_text_embedding_model(args.model_name)
    model.to(args.device)
    model.eval()

    # 3. Extract Embeddings
    save_dict = {}
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Text Embeddings"):
        text = row["caption"]
        if args.model_name == "bert-base-uncased":
            embedding = get_bert_embedding(model, tokenizer, text)
        elif args.model_name == "Qwen/Qwen3-Embedding-0.6B":
            instruction = "Retrieve music caption that matches the following video caption"
            embedding = get_qwen_embedding(model, tokenizer, text, instruction)
        else:
            raise ValueError(f"Unsupported text model: {args.model_name}")
        save_dict[row["id"]] = embedding

    # 4. Save
    model_name = args.model_name.split("/")[-1]
    save_path = f"cache/text_embedding/ossl/{model_name}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(save_dict, save_path)
    print(f"✅ Saved embeddings to: {save_path}")

if __name__ == "__main__":
    main()
