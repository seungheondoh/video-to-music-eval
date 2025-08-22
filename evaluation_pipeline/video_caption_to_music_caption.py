import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # --- Configuration ---
    # File paths
    INPUT_CAPTIONS_PATH = "/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/video_captions/video_captions_qwen.json"
    OUTPUT_CAPTIONS_PATH = "/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/video_captions/video_to_music_captions_qwen.json"

    # Model and device settings
    MODEL_ID = "Qwen/Qwen3-0.6B"
    DEVICE = "cuda" 

    # Generation parameters
    MAX_RETRIES = 4
    MIN_TOKENS = 20
    MAX_TOKENS = 300
    MAX_NEW_TOKENS_GENERATION = 1000

    # --- System Prompt ---
    # This prompt instructs the model on how to generate the music caption.
    SYSTEM_PROMPT_TEMPLATE = """You are given a video caption that describes the content of a video. 
    Imagine that the video already has background music that perfectly fits its mood, theme, and atmosphere. 
    Based on the video description, generate a text caption that describes what that music would be like. 
    The output must be only the music caption — do not include any explanations, formatting, or additional text.
    Be between {min_tokens} and {max_tokens} tokens long (do not go under or over this range).

    Video caption: {video_caption}
    Music caption:"""

    # --- Main Script ---

    # Load the model and tokenizer
    print(f"Loading model: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
    print("Model loaded successfully.")

    # Load existing results to resume processing if the script was interrupted
    try:
        with open(OUTPUT_CAPTIONS_PATH, "r", encoding="utf-8") as f:
            processed_results = json.load(f)
            processed_video_paths = set(item["video_path"] for item in processed_results)
    except FileNotFoundError:
        processed_results = []
        processed_video_paths = set()
    print(f"Found {len(processed_results)} previously processed captions.")

    # Load the input JSON file with video captions
    with open(INPUT_CAPTIONS_PATH, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # Process each video caption
    for item in tqdm(input_data, desc="Processing video captions"):
        video_path = item.get("video_path")
        video_caption = item.get("caption")

        if not video_path or not video_caption:
            print(f"Skipping item with missing video_path or caption: {item}")
            continue

        # Skip if this video has already been processed
        if video_path in processed_video_paths:
            continue

        # Format the prompt with the current video caption
        prompt_content = SYSTEM_PROMPT_TEMPLATE.format(
            min_tokens=MIN_TOKENS,
            max_tokens=MAX_TOKENS,
            video_caption=video_caption
        )
        
        messages = [{"role": "user", "content": prompt_content}]

        try:
            generated_music_caption = ""

            # Retry generation up to MAX_RETRIES times to meet token constraints
            for attempt in range(MAX_RETRIES):
                print(f"\nProcessing {os.path.basename(video_path)} - Attempt {attempt + 1}/{MAX_RETRIES}")

                # Prepare inputs for the model
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(DEVICE)

                # Generate text
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS_GENERATION,
                    do_sample=True,
                )

                # Decode the generated output
                full_output = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[-1]:],
                    skip_special_tokens=True
                ).strip()

                # Clean the output to get only the music caption
                if "</think>" in full_output:
                    generated_music_caption = full_output.split("</think>", 1)[-1].replace("Music caption:", "").replace("music caption:", "").strip()
                else:
                    generated_music_caption = full_output

                # Validate the token count of the generated caption
                num_tokens = len(tokenizer.encode(generated_music_caption))
                print(f"💬 Generated caption token count: {num_tokens}")

                # Check if the caption is valid (within token range and doesn't contain "video")
                is_valid_length = MIN_TOKENS <= num_tokens <= MAX_TOKENS
                contains_forbidden_word = "video" in generated_music_caption.lower()

                if is_valid_length and not contains_forbidden_word:
                    print("✅ Caption is valid.")
                    break  # Exit the retry loop
                else:
                    if not is_valid_length:
                        print(f"⚠️ Caption length ({num_tokens}) is outside the {MIN_TOKENS}-{MAX_TOKENS} token range.")
                    if contains_forbidden_word:
                        print("⚠️ Caption contains the forbidden word 'video'.")
                    
                    if attempt < MAX_RETRIES - 1:
                        print("Retrying...")
                    else:
                        print("⚠️ Max retries reached. Using the last generated caption anyway.")

            # Store the result
            result_item = {
                "video_path": video_path,
                "caption": generated_music_caption
            }
            processed_results.append(result_item)

            # Save results to the output file after each item is processed
            with open(OUTPUT_CAPTIONS_PATH, "w", encoding="utf-8") as f:
                json.dump(processed_results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"❌ Error processing {video_path}: {e}")

    print(f"\n✅ Processing complete. Results saved to: {OUTPUT_CAPTIONS_PATH}")

if __name__ == "__main__":
    main()