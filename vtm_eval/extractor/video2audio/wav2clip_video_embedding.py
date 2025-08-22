# -*- coding: utf-8 -*-
"""
This script extracts video embeddings from a directory of video files using a
pre-trained CLIP vision model. It processes each video by sampling one frame per
second, generating an embedding for each sampled frame, and then averaging these
embeddings to create a single representative embedding for the entire video.
The resulting embeddings are saved to a PyTorch tensor file.
"""

import os
import argparse
from glob import glob
import torch
import numpy as np
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection, AutoImageProcessor
from torchcodec.decoders import VideoDecoder

def load_vision_embedding_model():
    """
    Loads the pre-trained CLIP vision model and its associated image processor.

    Returns:
        tuple: A tuple containing the model and the processor, both ready for use.
    """
    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model.eval(), processor

def get_video_embedding(video_path, model, processor, device):
    """
    Generates a single embedding for a given video file.

    The function samples one frame per second from the video, processes these
    frames, and computes their embeddings using the provided CLIP model.
    The final video embedding is the mean of the individual frame embeddings.

    Args:
        video_path (str): The path to the video file.
        model (torch.nn.Module): The pre-trained vision model.
        processor: The image processor for the model.
        device (torch.device): The device (CPU or CUDA) to run inference on.

    Returns:
        torch.Tensor: A 1D tensor representing the video embedding, or None if failed.
    """
    try:
        video_decoder = VideoDecoder(video_path)
        metadata = video_decoder.metadata
        num_frames = metadata.num_frames
        average_fps = metadata.average_fps
        duration_seconds = metadata.duration_seconds

        # Sample one frame per second
        num_seconds = int(np.floor(duration_seconds))
        frame_indices = (np.arange(num_seconds) * int(round(average_fps))).clip(0, num_frames - 1)
        
        # Decode the sampled frames
        video_frames = video_decoder.get_frames_at(indices=frame_indices).data  # T x C x H x W

        # Preprocess frames and move to the specified device
        inputs = processor(images=video_frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        # Perform inference
        with torch.no_grad():
            model_output = model(pixel_values=pixel_values)
            # Average the embeddings of all sampled frames to get a single video embedding
            video_embedding = model_output.image_embeds.mean(dim=0, keepdim=True).cpu()  # (1, 512)

        return video_embedding.squeeze(0) # Return as (512,)
    except Exception as e:
        print(f"⚠️  Error processing {video_path}: {e}")
        return None

def main(args):
    """
    Main function to orchestrate the video embedding extraction process.
    """
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_dir = args.video_dir
    output_path = args.output_path

    # Load model and processor
    model, processor = load_vision_embedding_model()
    model.to(device)

    # Find all video files
    video_files = sorted(glob(os.path.join(video_dir, "*.mp4")))
    embedding_dict = {}

    # Process each video file
    for video_path in tqdm(video_files, desc="Extracting video embeddings"):
        embedding = get_video_embedding(video_path, model, processor, device)
        if embedding is not None:
            embedding_dict[video_path] = embedding
        else:
            print(f"⚠️  Failed to generate embedding for: {video_path}")

    # Save the embeddings to a file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(embedding_dict, output_path)
    print(f"✅ Saved {len(embedding_dict)} video embeddings to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract video embeddings using CLIP.")
    parser.add_argument(
        "--video_dir",
        type=str,
        default="/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/video_db",
        help="Directory containing video files."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/video_embeddings/wav2clip.pt",
        help="Path to save the output PyTorch tensor file."
    )
    args = parser.parse_args()
    main(args)
