import os
import numpy as np
import cv2

# 整理後の画像が保存されているベースディレクトリ（各動画フォルダ内に各トラックフォルダがある）
base_img_dir = "../../output/jersey-number-pipeline_outputs/OutdoorResults/ID_merging/crops/imgs/"
# ヒストグラムの保存先ディレクトリ
output_dir = "../../color_histograms/Outdoor/ID_merging/"
os.makedirs(output_dir, exist_ok=True)

# 各動画フォルダに対して処理
for video_folder in os.listdir(base_img_dir):
    video_path = os.path.join(base_img_dir, video_folder)
    if not os.path.isdir(video_path):
        continue

    # 出力用の動画フォルダを作成
    video_output_dir = os.path.join(output_dir, video_folder)
    os.makedirs(video_output_dir, exist_ok=True)

    # 各トラックフォルダに対して処理
    for track_folder in os.listdir(video_path):
        track_path = os.path.join(video_path, track_folder)
        if not os.path.isdir(track_path):
            continue

        # トラック内のpng画像リストを取得し、ファイル名でソート（順番が重要な場合）
        image_files = sorted([f for f in os.listdir(track_path) if f.lower().endswith('.png')])
        # 最初の100フレーム（画像）だけを使用
        image_files = image_files[:100]
        
        histograms = []
        for image_file in image_files:
            image_path = os.path.join(track_path, image_file)
            # OpenCVで画像読み込み（BGR形式）
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: {image_path}の読み込みに失敗しました。")
                continue

            # 3Dカラーヒストグラムの計算（B,G,Rそれぞれ8ビン、全512要素）
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = hist.flatten()  # 1次元配列に変換
            histograms.append(hist)

        # 対象画像が存在しなければスキップ
        if len(histograms) == 0:
            continue

        # 各画像のヒストグラムをスタックして、各ビンごとの中央値を計算
        hist_array = np.stack(histograms, axis=0)  # shape: (num_images, 512)
        median_hist = np.median(hist_array, axis=0)

        # 出力ファイル名に動画名とトラック名を含める例（例: 0107_1_track1.npy）
        output_file = os.path.join(video_output_dir, f"{video_folder}_{track_folder}.npy")
        np.save(output_file, median_hist)
        print(f"Saved histogram for {video_folder}/{track_folder} to {output_file}")

print("Finished.")