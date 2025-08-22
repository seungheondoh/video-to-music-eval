## Code Explanation

### DB Preparation
- `generate_audio_caption.py`: Generating Pseudo-Captions from Audio to Build Database

### Text2Text Retrieval
- `generate_video_caption.py`: Generate Video Caption
- `video_caption_to_music_caption.py`: Video Caption -> Music Caption
- `video_text_embedding.py`: Video Caption -> Video Text Embedding
- `music_text_embedding.py`: Music Caption -> Music Text Embedding

### Video2Audio Retrieval
- `audioclip_inference.py`: Generate Audio(Music) Embedding with AudioCLIP
- `audioclip_video_inference.py`: Generate Video Embedding with AudioCLIP
- `wav2clip_inference.py`: Generate Audio(Music) Embedding with Wav2CLIP
- `wav2clip_video_inference.py`: Generate Video Embedding with OpenAI/CLIP

### Evaluation
- `calculate_similarity.py`: Retrieve top K from Embeddings (Save Json)
- `evaluate.py`: Calculate Recall@K and Median Rank from Json

### Execution
```
python run.py --steps STEPS_TO_EXECUTE
```
