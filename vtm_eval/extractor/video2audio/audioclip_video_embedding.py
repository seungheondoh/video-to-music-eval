import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm
import torchvision.transforms as tv

# Constants for image preprocessing
IMAGE_SIZE = 224
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

def setup_audioclip_path(audioclip_project_path):
    """Add the AudioCLIP project directory to the system path."""
    if audioclip_project_path not in sys.path:
        sys.path.append(audioclip_project_path)

def load_audioclip_model(model_path, device="cuda:0"):
    """
    Loads the AudioCLIP model from a specified path.
    """
    from model import AudioCLIP
    model = AudioCLIP(pretrained=model_path)
    model.eval().to(device)
    return model

@torch.no_grad()
def extract_videoclip_embedding(model, video_path, device="cuda:0"):
    """
    Extracts a video embedding using the AudioCLIP model.
    It samples one frame per second, computes the image embedding for each,
    and then averages them to get a single video-level embedding.
    """
    # VideoDecoder extracts frames on the CPU
    try:
        decoder = VideoDecoder(video_path, device="cpu")
    except Exception as e:
        print(f"Error opening video file {video_path}: {e}")
        return None

    num_frames = decoder.metadata.num_frames
    average_fps = decoder.metadata.average_fps
    duration_seconds = decoder.metadata.duration_seconds

    if num_frames == 0 or average_fps is None or duration_seconds is None:
        print(f"Could not read metadata from video: {video_path}")
        return None

    num_seconds = int(np.floor(duration_seconds))
    if num_seconds == 0:
        return None
        
    frame_indices = (np.arange(num_seconds) * int(round(average_fps))).clip(0, num_frames - 1)
    
    frames_tensor = decoder.get_frames_at(indices=frame_indices).data  # (T, C, H, W)

    preprocess = tv.Compose([
        tv.Resize(IMAGE_SIZE, interpolation=Image.BICUBIC),
        tv.CenterCrop(IMAGE_SIZE),
        tv.Normalize(IMAGE_MEAN, IMAGE_STD)
    ])

    embeddings = []
    for frame in frames_tensor:
        # The frame from torchcodec is already a tensor [C, H, W] with values in [0, 255]
        # We need to convert it to float and scale to [0, 1] for preprocessing.
        tensor = frame.float() / 255.0
        tensor = preprocess(tensor)
        tensor = tensor.unsqueeze(0).to(device)
        
        image_feature = model.encode_image(tensor)
        image_feature = F.normalize(image_feature, dim=-1)
        embeddings.append(image_feature.squeeze(0).cpu())

    if embeddings:
        video_emb = torch.stack(embeddings).mean(dim=0)
        video_emb = F.normalize(video_emb, dim=0)
        return video_emb
    else:
        return None

def main(args):
    """
    Main function to extract video embeddings for all videos in a directory.
    """
    setup_audioclip_path(args.audioclip_project_path)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_audioclip_model(args.model_path, device=device)

    video_files = sorted([
        os.path.join(args.video_dir, f) 
        for f in os.listdir(args.video_dir) 
        if f.lower().endswith((".mp4"))
    ])
    
    embedding_dict = {}

    for video_path in tqdm(video_files, desc="Extracting video embeddings"):
        emb = extract_videoclip_embedding(model, video_path, device)
        if emb is not None:
            video_filename = os.path.basename(video_path)
            embedding_dict[video_filename] = emb
        else:
            print(f"⚠️ Failed to create embedding for: {video_path}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(embedding_dict, args.output_path)
    print(f"✅ Saved {len(embedding_dict)} video embeddings to: {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract video embeddings using AudioCLIP.")
    
    parser.add_argument(
        "--video_dir", 
        type=str, 
        required=True, 
        help="Directory containing video files."
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True, 
        help="Path to save the output .pt file with embeddings."
    )
    parser.add_argument(
        "--audioclip_project_path", 
        type=str, 
        required=True, 
        help="Path to the root of the AudioCLIP project directory."
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the pretrained AudioCLIP model file (.pt)."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda", 
        help="Device to use for inference (e.g., 'cuda', 'cpu')."
    )

    args = parser.parse_args()
    main(args)
