import os
import re
import shutil

# 画像が保存されているディレクトリのパス
src_dir = "../../output/jersey-number-pipeline_outputs/OutdoorResults/ID_merging/crops/imgs/"

# ファイル名のパターン例: IMG_0107_1_frame0_track1_cropped.png
pattern = re.compile(r'IMG_(\d+_\d+)_frame\d+_track(\d+)_cropped\.png')

print(f"Starting to organize images in: {src_dir}")

# src_dir内の全ファイルを走査
for idx, filename in enumerate(os.listdir(src_dir), start=1):
    # 進捗表示
    print(f"[{idx}] Checking file: {filename}")
    if not filename.endswith(".png"):
        print("--> Skipped (not a PNG)")
        continue
    match = pattern.match(filename)
    if match:
        video_id = match.group(1)  # 例: "0107_1"
        track_num = match.group(2)  # 例: "1"
        print(f"--> Match found: video_id={video_id}, track_num={track_num}")
        
        # 各動画ごとのフォルダ、さらにトラックごとのフォルダを作成
        target_dir = os.path.join(src_dir, video_id, f"track{track_num}")
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
            print(f"----> Created directory: {target_dir}")
        else:
            print(f"----> Directory already exists: {target_dir}")
        
        # ファイルの移動
        src_path = os.path.join(src_dir, filename)
        target_path = os.path.join(target_dir, filename)
        try:
            shutil.move(src_path, target_path)
            print(f"----> Moved file to: {target_path}\n")
        except Exception as e:
            print(f"!!!! Error moving file {filename}: {e}\n")
    else:
        print("--> No match for pattern, skipped.\n")

print("Organizing complete.")
