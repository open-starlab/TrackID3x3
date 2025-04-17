import os
import re
import shutil

# 画像が保存されているディレクトリのパス
src_dir = "out/OutdoorResults/crops/imgs"

# ファイル名のパターン例: IMG_0107_1_frame0_track1_cropped.png
pattern = re.compile(r'IMG_(\d+_\d+)_frame\d+_track(\d+)_cropped\.png')

# src_dir内の全ファイルを走査
for filename in os.listdir(src_dir):
    if not filename.endswith(".png"):
        continue
    match = pattern.match(filename)
    if match:
        video_id = match.group(1)  # 例: "0107_1"
        track_num = match.group(2)  # 例: "1"
        
        # 各動画ごとのフォルダ、さらにトラックごとのフォルダを作成
        target_dir = os.path.join(src_dir, video_id, f"track{track_num}")
        os.makedirs(target_dir, exist_ok=True)
        
        # ファイルの移動
        src_path = os.path.join(src_dir, filename)
        target_path = os.path.join(target_dir, filename)
        shutil.move(src_path, target_path)
