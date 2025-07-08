import json
import pandas as pd
import os

json_path = "/home/daeyong/gaudio_retrieval_evaluation/video-to-music-eval/vtm_eval/inference/video2audio/results/video_music_retrieval_results_wav2clip.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df['video_id'] = sorted(os.listdir("/home/daeyong/gaudio_retrieval_evaluation/ossl/video"))[:len(df)]

def evaluate_retrieval(df, ks=[1, 5, 10, 50, 100]):
	recalls = {k: 0 for k in ks}
	ranks = []

	for idx, row in df.iterrows():
		# 정답 ID: '0.mp4' -> '0'
		gt_id = row['video_id'].replace('.mp4', '')
		# print(f"🔍 Evaluating video: {row['video_id']}")
		retrieved_ids = [item['music_id'] for item in row['retrieval_results']]

		if gt_id in retrieved_ids:
			rank = retrieved_ids.index(gt_id) + 1  # 1-based index
			ranks.append(rank)

			for k in ks:
				if rank <= k:
					recalls[k] += 1
		else:
			ranks.append(201)  # out of top 200

	total = len(df)

	# 결과 정리
	recall_at_k = {f"Recall@{k}": round(recalls[k] / total, 4) for k in ks}
	median_rank = float(pd.Series(ranks).median())

	return recall_at_k, median_rank, ranks

recall_results, med_rank, all_ranks = evaluate_retrieval(df)
print("🔹 Recall@K:", recall_results)
print("🔹 Median Rank:", med_rank)

print(len([r for r in all_ranks if r <= 200]), "videos found in top 200")