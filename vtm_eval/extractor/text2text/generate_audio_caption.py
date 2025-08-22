import os
import json
import torch
import subprocess
from time import time
from pathlib import Path
from tqdm import tqdm
from transformers import BitsAndBytesConfig, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import argparse

# --- Constants ---
SYSTEM_PROMPT = (
    "You are an expert music captioner."
    "Write a detailed, third-person description of the audio only."
    "Describe the genre, key instruments/timbre, mood, tempo/groove, and any notable production traits (e.g., reverb, distortion, synth texture)."
    "Do NOT ask questions. Do NOT address the listener. Do NOT use first or second person."
    "Do NOT include meta commentary, opinions about the listener, or suggestions."
    "Only output the description."
)

def make_audio_proxy(src_path: str, max_sec: int = 120, sr: int = 16000, mono: bool = True) -> str:
    """
    Converts an audio file into a standardized proxy format using ffmpeg.

    This function preprocesses audio by:
    - Truncating it to a maximum duration (max_sec).
    - Resampling to a target sample rate (sr).
    - Downmixing to mono.

    Args:
        src_path: Path to the source audio file.
        max_sec: Maximum duration of the output audio in seconds.
        sr: Target sample rate in Hz.
        mono: If True, convert audio to mono.

    Returns:
        The path to the temporary proxy WAV file. If conversion fails,
        returns the original source path.
    """
    base_name = Path(src_path).stem
    out_path = f"/tmp/{base_name}_proxy.wav"
    
    command = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-t", str(max_sec),
        "-ac", "1" if mono else "2",
        "-ar", str(sr),
        "-vn",  # No video
        "-sample_fmt", "s16",  # 16-bit PCM
        out_path
    ]
    
    try:
        # Run ffmpeg, suppressing output for cleaner logs
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if os.path.exists(out_path):
            return out_path
    except Exception:
        # If ffmpeg fails, fall back to the original path
        pass
    return src_path


@torch.no_grad()
def get_audio_caption_qwen(
    caption_model: Qwen2_5OmniForConditionalGeneration,
    caption_processor: Qwen2_5OmniProcessor,
    audio_path: str
) -> str:
    """
    Generates a music caption for a given audio file using the Qwen2.5-Omni model.

    Args:
        caption_model: The loaded Qwen model for generation.
        caption_processor: The processor for the Qwen model.
        audio_path: Path to the input audio file (.wav).

    Returns:
        A string containing the generated music caption.
    """
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "audio", "audio": audio_path}]},
    ]

    # Apply chat template to create the text prompt
    text_prompt = caption_processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False
    )

    # Preprocess multimodal inputs (in this case, only audio)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = caption_processor(
        text=text_prompt,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True
    )
    inputs = inputs.to(caption_model.device, dtype=caption_model.dtype)

    # Generate caption
    output_ids = caption_model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )

    # Decode the generated tokens
    text_out = caption_processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    # Extract the assistant's response
    caption = text_out[0].split("assistant\n")[-1]
    return caption.strip()


def main(args: argparse.Namespace):
    """
    Main function to generate audio captions for a dataset of audio files.
    """
    # --- Model Loading ---
    print("🔧 Loading Qwen2.5-Omni model...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    caption_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="auto"
    )
    caption_model.disable_talker()
    caption_processor = Qwen2_5OmniProcessor.from_pretrained(args.model_name)

    # --- Setup Output ---
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file = output_dir / "audio_captions_qwen.json"

    # --- Load Existing Results to Resume ---
    if save_file.exists():
        with open(save_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        existing_audio_paths = {item["audio_path"] for item in results}
        print(f"📄 Loaded {len(results)} existing captions. Resuming...")
    else:
        results = []
        existing_audio_paths = set()

    # --- Process Audio Files ---
    audio_files = sorted([
        str(p) for p in Path(args.dataset_path).glob("*.wav")
    ])

    print("🎵 Starting audio caption generation...")
    for audio_path in tqdm(audio_files, desc="Processing audios"):
        if audio_path in existing_audio_paths:
            print(f"⏩ Skipping (already processed): {audio_path}")
            continue

        try:
            start_time = time()
            
            # Create a safe, standardized audio proxy for the model
            safe_audio_path = make_audio_proxy(audio_path, max_sec=120, sr=16000, mono=True)
            
            caption = get_audio_caption_qwen(caption_model, caption_processor, safe_audio_path)
            
            elapsed_time = round(time() - start_time, 2)

            results.append({
                "audio_path": audio_path,
                "caption": caption,
                "elapsed_time_sec": elapsed_time,
            })

            # Save progress after each file
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
            # Clean up temporary proxy file
            if safe_audio_path.startswith("/tmp/") and os.path.exists(safe_audio_path):
                os.remove(safe_audio_path)

        except Exception as e:
            print(f"❌ Failed to process {audio_path}: {e}")

    print(f"\n✅ Saved {len(results)} total captions to {save_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate audio captions using the Qwen2.5-Omni model."
    )
    parser.add_argument(
        "--dataset_path", type=str,
        default="/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/audio_db",
        help="Path to the directory containing audio files (.wav)."
    )
    parser.add_argument(
        "--output_path", type=str,
        default="/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/audio_captions",
        help="Path to the directory where the output JSON file will be saved."
    )
    parser.add_argument(
        "--model_name", type=str,
        default="Qwen/Qwen2.5-Omni-7B",
        help="Name of the Hugging Face model to use."
    )
    args = parser.parse_args()

    main(args)
