import os
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModel
import ast
import torch
import argparse
from tqdm import tqdm
from torch import Tensor

def load_text_embedding_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer

def get_bert_embedding(model, tokenizer, text):
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    outputs = outputs.last_hidden_state.mean(dim=1)
    outputs = outputs.detach().cpu()
    return outputs

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

def get_qwen_embedding(model, tokenizer, text, instruction=None):
    if instruction is not None:
        text = get_detailed_instruct(instruction, text)
    batch_dict = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="pt",
    )
    batch_dict.to(model.device)
    with torch.no_grad():
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = embeddings.detach().cpu()
    return embeddings
