# -*- coding: utf-8 -*-
"""
This script generates detailed video captions using the Qwen2.5-Omni model.

It processes a directory of video files, creates standardized proxy videos
to ensure consistent VRAM usage, generates captions using a multimodal LLM,
and saves the results to a JSON file. The script is designed to be
resumable, skipping videos that have already been processed.
"""

import argparse
import gc
import json
import os
import subprocess
import uuid
from pathlib import Path
from time import time

import torch
from tqdm import tqdm
from transformers import (BitsAndBytesConfig,
                          Qwen2_5OmniForConditionalGeneration,
                          Qwen2_5OmniProcessor)

from qwen_omni_utils import process_mm_info

# Configure PyTorch CUDA memory allocator to be more flexible.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -----------------------------------------------------------------------------
# Video Processing Utilities
# -----------------------------------------------------------------------------

def get_video_stats(video_path: str) -> dict | None:
    """
    Extracts detailed video metadata using ffprobe.

    Args:
        video_path: The path to the video file.

    Returns:
        A dictionary containing video statistics (width, height, duration, fps,
        frame count, bitrate, size) or None if an error occurs.
    """
    if not os.path.exists(video_path):
        print(f"⚠️ Video file not found: {video_path}")
        return None
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,duration,nb_frames",
            "-show_entries", "format=bit_rate",
            "-of", "json",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        stream_data = data.get("streams", [{}])[0]
        format_data = data.get("format", {})

        # Parse frame rate which can be a fraction (e.g., "30/1").
        frame_rate_str = stream_data.get("avg_frame_rate", "0/1")
        if "/" in frame_rate_str:
            num, den = map(int, frame_rate_str.split('/'))
            fps = float(num / den) if den != 0 else 0.0
        else:
            fps = float(frame_rate_str)

        return {
            "width": stream_data.get("width"),
            "height": stream_data.get("height"),
            "duration_sec": float(stream_data.get("duration", 0)),
            "fps": round(fps, 2),
            "total_frames": int(stream_data.get("nb_frames", 0)),
            "bitrate_kbps": int(format_data.get("bit_rate", 0)) // 1000,
            "size_mb": round(os.path.getsize(video_path) / (1024 * 1024), 2)
        }
    except Exception as e:
        print(f"⚠️ Failed to get stats for {video_path}: {e}")
        return None


def _probe_duration_sec(path: str) -> float | None:
    """
    A lightweight function to get only the video duration in seconds.

    Args:
        path: The path to the video file.

    Returns:
        The duration in seconds as a float, or None on failure.
    """
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", path
        ]
        out = subprocess.check_output(cmd)
        duration = float(json.loads(out)["format"]["duration"])
        return duration
    except Exception:
        return None


def extract_frames_as_video(
    src_path: str,
    target_total_frames: int = 16,
    short_side: int = 224,
    max_sec: int = 90
) -> str | None:
    """
    Creates a new video clip by uniformly sampling frames from the source.

    This method uses ffmpeg's 'select' filter to pick frames at a fixed
    interval, which is useful for creating a representative summary of a video.

    Args:
        src_path: Path to the source video.
        target_total_frames: The desired number of frames in the output clip.
        short_side: The target resolution for the shorter side of the video.
        max_sec: The maximum duration of the source video to process.

    Returns:
        The path to the generated proxy video, or None on failure.
    """
    stats = get_video_stats(src_path)
    if not stats or stats.get("total_frames", 0) == 0:
        print(f"⚠️ Could not get frame count for {src_path}, skipping.")
        return None

    original_fps = stats.get("fps", 30)
    duration = stats["duration_sec"]

    clip_len = min(duration, max_sec)
    frames_to_process = int(clip_len * original_fps)

    if frames_to_process <= 0:
        print(f"⚠️ No frames to process for {src_path}, skipping.")
        return None

    # Calculate the interval to select one frame every N frames.
    interval = max(1, round(frames_to_process / target_total_frames))

    base_name = Path(src_path).stem
    extension = os.path.splitext(src_path)[1]
    out_path = f"/tmp/{base_name}_proxy_{uuid.uuid4().hex[:8]}{extension}"

    # ffmpeg filter graph:
    # 1. select: Pick frames based on the calculated interval.
    # 2. scale: Resize while maintaining aspect ratio.
    # 3. setpts: Re-time the presentation timestamps for a smooth video.
    video_filter = (
        f"select='not(mod(n\\,{interval}))',"
        f"scale='if(gt(iw,ih),-2,{short_side})':'if(gt(iw,ih),{short_side},-2)',"
        f"setpts=N/FRAME_RATE/TB"
    )

    cmd = [
        "ffmpeg", "-y",
        "-t", str(clip_len),
        "-i", src_path,
        "-an",  # No audio
        "-vf", video_filter,
        "-c:v", "libx264",
        "-preset", "veryfast",
        out_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Frame extraction failed for {src_path}. Stderr: {result.stderr}")
        return None

    final_stats = get_video_stats(out_path)
    if final_stats:
        print(f"✅ Proxy created with {final_stats['total_frames']} frames.")

    return out_path if os.path.exists(out_path) else None


def make_video_proxy_final(
    src_path: str,
    target_total_frames: int = 16,
    short_side: int = 384,
    max_sec: int = 90
) -> str | None:
    """
    Standardizes all videos to a fixed total frame count to stabilize VRAM usage.

    This method dynamically calculates the required output FPS to ensure the
    final clip always contains `target_total_frames`.

    Args:
        src_path: Path to the source video.
        target_total_frames: The exact number of frames for the output video.
        short_side: The target resolution for the shorter side.
        max_sec: The maximum duration of the source video to process.

    Returns:
        The path to the generated proxy video, or None on failure.
    """
    duration = _probe_duration_sec(src_path)
    if not duration or duration <= 0:
        print(f"⚠️ Could not probe duration for {src_path}, skipping.")
        return None

    clip_len = min(duration, max_sec)

    # Core logic: Calculate output FPS by dividing target frames by clip duration.
    # e.g., 32 frames / 90 sec = 0.355 FPS; 32 frames / 10 sec = 3.2 FPS
    out_fps = target_total_frames / clip_len

    base_name = Path(src_path).stem
    extension = os.path.splitext(src_path)[1]
    out_path = f"/tmp/{base_name}_proxy_{uuid.uuid4().hex[:8]}{extension}"

    video_filter = (
        f"fps={out_fps},"
        f"scale='if(gt(iw,ih),-2,{short_side})':'if(gt(iw,ih),{short_side},-2)'"
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", "0", "-t", str(clip_len),  # Trim the source video
        "-i", src_path,
        "-an",  # No audio
        "-vf", video_filter,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "fastdecode",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        out_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Proxy video creation failed for {src_path}. Stderr: {result.stderr}")
        return None

    return out_path if os.path.exists(out_path) else None


# -----------------------------------------------------------------------------
# Qwen2.5 Omni Inference
# -----------------------------------------------------------------------------

@torch.no_grad()
def get_video_caption_qwen(
    caption_model: Qwen2_5OmniForConditionalGeneration,
    caption_processor: Qwen2_5OmniProcessor,
    video_path: str,
    use_audio: bool = False
) -> str:
    """
    Generates a caption for a given video using the Qwen-Omni model.

    Args:
        caption_model: The loaded Qwen-Omni model.
        caption_processor: The processor for the model.
        video_path: Path to the video file to be captioned.
        use_audio: Whether to include audio data in the model input.

    Returns:
        The generated video caption as a string.
    """
    # System prompt to guide the model's output.
    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a professional video captioning system. "
                        "Carefully watch the provided video frames. "
                        "Describe the scene in detail, covering:\n"
                        "- Key actions and events in chronological order\n"
                        "- People, objects, and environment\n"
                        "- Notable visual changes over time\n"
                        "- Colors, lighting, and atmosphere\n"
                        "Do not ask questions or provide commentary beyond the description."
                    )
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path}
            ],
        },
    ]

    # Apply chat template to format the text prompt.
    text_prompt = caption_processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False
    )

    # Preprocess multimodal inputs (video, audio, images).
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio)
    inputs = caption_processor(
        text=text_prompt,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio
    )

    inputs = inputs.to(caption_model.device, dtype=caption_model.dtype)

    # Generate token IDs for the caption.
    output_ids = caption_model.generate(
        **inputs,
        use_audio_in_video=use_audio,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )

    # Decode the output and clean up the text.
    text_out = caption_processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    # Extract the assistant's response and remove boilerplate.
    caption = text_out[0].split("assistant\n")[-1]
    caption = caption.split("What do you think")[0].split("what do you think")[0]
    return caption.strip()

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main(args):
    """
    Main function to run the video captioning pipeline.
    """
    # --- 1. Load Model ---
    print("🔧 Loading Qwen2.5-Omni model...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # Define memory limits for model loading across devices.
    max_memory = {
        0: "10GiB",
        1: "10GiB",
        "cpu": "50GiB"
    }

    caption_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="auto",
        max_memory=max_memory,
    )
    caption_model.disable_talker()  # Suppress verbose model outputs.
    caption_processor = Qwen2_5OmniProcessor.from_pretrained(args.model_name)

    # --- 2. Prepare Output and Load Existing Results ---
    os.makedirs(args.output_path, exist_ok=True)
    output_json_path = os.path.join(args.output_path, "video_captions_qwen.json")

    if os.path.exists(output_json_path):
        with open(output_json_path, "r", encoding="utf-8") as f:
            caption_results = json.load(f)
        existing_video_paths = {item["video_path"] for item in caption_results}
        print(f"📄 Loaded {len(caption_results)} existing captions. Resuming...")
    else:
        caption_results = []
        existing_video_paths = set()

    # --- 3. Process Videos ---
    video_files = sorted([
        os.path.join(args.dataset_path, f)
        for f in os.listdir(args.dataset_path)
        if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
    ])

    print(f"📽️ Starting caption generation for {len(video_files)} videos...")
    for video_path in tqdm(video_files, desc="Processing videos"):
        if video_path in existing_video_paths:
            tqdm.write(f"⏩ Skipping (already processed): {Path(video_path).name}")
            continue

        tqdm.write(f"\n▶️ Processing: {Path(video_path).name}")
        proxy_path = None  # Initialize proxy path for cleanup

        try:
            # --- 3a. Create Proxy Video ---
            # Using `extract_frames_as_video` to sample a fixed number of frames.
            # This helps stabilize processing time and VRAM usage.
            proxy_path = extract_frames_as_video(
                video_path,
                target_total_frames=16,
                short_side=320,
                max_sec=500  # Use a large value to process the full video length
            )

            if not proxy_path:
                tqdm.write(f"❌ Failed to create proxy for {video_path}. Skipping.")
                continue

            # --- 3b. Generate Caption ---
            start_time = time()
            caption = get_video_caption_qwen(
                caption_model,
                caption_processor,
                proxy_path,
                use_audio=False
            )
            elapsed_time = round(time() - start_time, 2)
            tqdm.write(f"💬 Caption generated in {elapsed_time}s: \"{caption[:80]}...\"")

            # --- 3c. Save Result ---
            result = {
                "video_path": video_path,
                "caption": caption,
                "elapsed_time_sec": elapsed_time,
            }
            caption_results.append(result)

            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(caption_results, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"❌ An unexpected error occurred while processing {video_path}: {e}")

        finally:
            # --- 3d. Cleanup ---
            if proxy_path and os.path.exists(proxy_path):
                try:
                    Path(proxy_path).unlink()
                    tqdm.write(f"🧹 Removed temp proxy: {proxy_path}")
                except OSError as e:
                    tqdm.write(f"⚠️ Failed to delete temp proxy {proxy_path}: {e}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    print(f"\n✅ Finished. Saved {len(caption_results)} total captions to {output_json_path}")

# -----------------------------------------------------------------------------
# Script Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate video captions using Qwen2.5-Omni."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/video_db",
        help="Path to the directory containing video files."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/video_captions",
        help="Path to the directory where the output JSON file will be saved."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-Omni-7B",
        help="The Hugging Face model identifier for Qwen-Omni."
    )
    args = parser.parse_args()

    main(args)
