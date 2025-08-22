import argparse
import subprocess
from types import SimpleNamespace

from quantitative_eval.evaluation_pipeline.extractor.text2text.generate_video_caption import main as generate_video_caption_main
from quantitative_eval.evaluation_pipeline.extractor.text2text.video_caption_to_music_caption import main as video_caption_to_music_caption_main
from quantitative_eval.evaluation_pipeline.extractor.text2text.video_text_embedding import main as video_text_embedding_main
from quantitative_eval.evaluation_pipeline.extractor.text2text.music_text_embedding import main as music_text_embedding_main
from quantitative_eval.evaluation_pipeline.audioclip_audio_embedding import main as audioclip_inference_main
from quantitative_eval.evaluation_pipeline.audioclip_video_embedding import main as audioclip_video_inference_main
from quantitative_eval.evaluation_pipeline.wav2clip_audio_embedding import main as wav2clip_inference_main
from quantitative_eval.evaluation_pipeline.wav2clip_video_embedding import main as wav2clip_video_inference_main
from quantitative_eval.evaluation_pipeline.inference.inference import main as calculate_similarity_main
from quantitative_eval.evaluation_pipeline.evaluate.evaluation import main as evaluation_main
from quantitative_eval.vtm_eval.extractor.text2text.generate_audio_caption import main as generate_audio_caption_main

def run_command(command):
    """Runs the given command in a subprocess."""
    print(f"Running command: {' '.join(command)}")
    subprocess.run(command, check=True)

def main(args):
    run_all = not args.steps

    # 1. Generate Captions
    if run_all or 'captions' in args.steps:
        print("\n--- 1. Generating Video Captions ---")
        # 1a. Video -> Video Caption
        print("\n[1a] Generating video captions...")
        generate_video_caption_main(SimpleNamespace(
            dataset_path="/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/video_db",
            output_path="/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/video_captions",
            model_name="Qwen/Qwen2.5-Omni-7B"
        ))
        
        # 1b. Video Caption -> Music Caption
        print("\n[1b] Converting video captions to music captions...")
        video_caption_to_music_caption_main()

    # 1-2. Generate Audio Captions
    if run_all or 'audio_captions' in args.steps:
        # 1c. Audio -> Audio Caption (for DB)
        print("\n--- 1-2. Generating Audio Captions ---")
        print("\n[1c] Generating audio captions for DB...")
        generate_audio_caption_main(SimpleNamespace(
            dataset_path="/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/audio_db",
            output_path="/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/audio_captions",
            model_name="Qwen/Qwen2.5-Omni-7B"
        ))

    # 2. Generate Embeddings
    if run_all or 'embeddings' in args.steps:
        print("\n--- 2. Generating Embeddings ---")
        
        # 2a. Text Embedding
        print("\n[2a] Generating text embeddings...")
        # Video Caption -> Embedding
        video_text_embedding_main(SimpleNamespace(
            json_path="/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/video_captions/video_captions_qwen.json",
            out_path="/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/video_embeddings/qwen.pt",
            device="cuda",
            batch_size=8
        ))
        # Music Caption -> Embedding
        video_text_embedding_main(SimpleNamespace(
            json_path="/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/video_captions/video_to_music_captions_qwen.json",
            out_path="/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/video_embeddings/qwen_music.pt",
            device="cuda",
            batch_size=8
        ))
        # Audio Caption -> Embedding
        music_text_embedding_main(SimpleNamespace(
            json_path="/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/audio_captions/audio_captions_qwen.json",
            out_path="/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/audio_embeddings/qwen.pt",
            device="cuda",
            batch_size=8
        ))

        # 2b. AudioCLIP Embedding
        print("\n[2b] Generating AudioCLIP embeddings...")
        audioclip_inference_main(SimpleNamespace(
            audio_root="/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/audio_db",
            output_path="/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/audio_embeddings/audioclip.pt",
            model_path="/home/daeyong/gaudio_retrieval_evaluation/audioclip/assets/AudioCLIP-Full-Training.pt",
            audioclip_module_path="/home/daeyong/gaudio_retrieval_evaluation/audioclip",
            device="cuda"
        ))
        audioclip_video_inference_main(SimpleNamespace(
            video_dir="/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/video_db",
            output_path="/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/video_embeddings/audioclip.pt",
            audioclip_project_path="/home/daeyong/gaudio_retrieval_evaluation/audioclip",
            model_path="/home/daeyong/gaudio_retrieval_evaluation/audioclip/assets/AudioCLIP-Full-Training.pt",
            device="cuda"
        ))

        # 2c. Wav2CLIP Embedding
        print("\n[2c] Generating Wav2CLIP embeddings...")
        wav2clip_inference_main(SimpleNamespace(
            audio_root="/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/audio_db",
            save_path="/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/audio_embeddings/wav2clip.pt"
        ))
        wav2clip_video_inference_main(SimpleNamespace(
            video_dir="/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/video_db",
            output_path="/home/daeyong/gaudio_retrieval_evaluation/quantitative_eval/video_embeddings/wav2clip.pt"
        ))

    # 3. Calculate Similarity
    if run_all or 'similarity' in args.steps:
        print("\n--- 3. Calculating Similarity ---")
        calculate_similarity_main()

    # 4. Run Evaluation
    if run_all or 'evaluation' in args.steps:
        print("\n--- 4. Running Evaluation ---")
        evaluation_main()

    print("\n--- Pipeline Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantitative evaluation pipeline runner.")
    parser.add_argument(
        '--steps', 
        nargs='+', 
        default=[], 
        help='Which steps to run. Default is all steps. Options: captions, audio_captions, embeddings, similarity, evaluation'
    )
    args = parser.parse_args()
    main(args)
