import os
import cv2
import random
import csv
import re

# パスの設定
video_folder = '../../videos/Outdoor/top/split'  # 分割済み動画が入っているフォルダ
mot_folder = '../../ground_truth/Outdoor/MOT_files'  # 元のMOT txtファイルが入っているフォルダ
csv_folder = '../../ground_truth/Outdoor/delimitation_frames'  # 区切りフレームを記述したcsvファイルが入っているフォルダ

# 出力フォルダの作成（bbox描画済み動画と分割txt用）
output_video_folder = os.path.join(video_folder, 'bbox_drawn')
os.makedirs(output_video_folder, exist_ok=True)
# 分割したtxtファイルの出力先として「split」フォルダを使用
output_txt_folder = os.path.join(mot_folder, 'split')
os.makedirs(output_txt_folder, exist_ok=True)

# MOT_files内のtxtファイルについて処理
for mot_file in os.listdir(mot_folder):
    if not mot_file.lower().endswith('.txt'):
        continue
    base_name = os.path.splitext(mot_file)[0]  # 例: IMG_0104
    mot_path = os.path.join(mot_folder, mot_file)
    csv_path = os.path.join(csv_folder, base_name + '.csv')
    if not os.path.exists(csv_path):
        print(f"[警告] {mot_file} に対応するcsvファイルが見つかりません。")
        continue

    # CSVファイルから各行のフレーム番号を抽出し、2行ごとに区切りを定義する
    boundaries = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            col = row[0].strip()  # 例: "117 (Game start)"
            m = re.match(r'(\d+)', col)
            if m:
                boundaries.append(int(m.group(1)))
    if len(boundaries) < 2:
        print(f"[警告] {csv_path} から十分な区切り情報が得られませんでした。")
        continue

    # グループ化：1行目～2行目、3行目～4行目、…（奇数行は無視されるか、もしくは最後のペアがない場合は警告）
    segments = []
    if len(boundaries) % 2 != 0:
        print(f"[警告] {csv_path} の行数が奇数です。最後の行は無視されます。")
    for i in range(0, len(boundaries) - 1, 2):
        start_frame = boundaries[i]
        end_frame = boundaries[i + 1]
        segments.append((start_frame, end_frame))

    # MOTのtxtファイル全体のbboxデータを読み込む
    # 各行の形式：frame番号, track_id, x, y, w, h
    mot_data = []
    with open(mot_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # カンマ区切りの場合（0始まりなので+1する）
            if ',' in line:
                parts = [p.strip() for p in line.split(',')]
                try:
                    frame_num = int(float(parts[0])) + 1
                except ValueError:
                    continue
            else:
                parts = line.split()
                try:
                    frame_num = int(float(parts[0]))
                except ValueError:
                    continue
            try:
                track_id = int(float(parts[1]))
                x, y, w, h = map(float, parts[2:6])
            except ValueError:
                continue
            mot_data.append((frame_num, track_id, int(x), int(y), int(w), int(h)))
    
    # 各セグメントごとに処理
    for idx, (seg_start, seg_end) in enumerate(segments, start=1):
        # セグメント内の MOT データ抽出（分割動画側は1始まりにリベース）
        segment_data = [
            (frame - seg_start + 1, track_id, x, y, w, h)
            for (frame, track_id, x, y, w, h) in mot_data
            if seg_start <= frame <= seg_end
        ]
        
        # 分割txtファイルとして保存（例: IMG_0104_1.txt）
        split_txt_name = f"{base_name}_{idx}.txt"
        split_txt_path = os.path.join(output_txt_folder, split_txt_name)
        with open(split_txt_path, 'w') as f_out:
            for entry in segment_data:
                f_out.write(" ".join(map(str, entry)) + "\n")
        
        # 分割済み動画のファイル名は「IMG_0104_1.mov」のようになっている前提
        video_file = f"{base_name}_{idx}.MOV"  # 拡張子の大文字・小文字は環境に合わせて調整
        video_path = os.path.join(video_folder, video_file)
        if not os.path.exists(video_path):
            print(f"[警告] 分割動画 {video_file} が見つかりません。")
            continue
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[エラー] {video_file} を開けませんでした。")
            continue
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = os.path.join(output_video_folder, video_file)
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # セグメント動画用に、フレーム番号（1始まり）をキーとするbbox辞書を作成
        bbox_data = {}
        for (frame, track_id, x, y, w, h) in segment_data:
            bbox_data.setdefault(frame, []).append((track_id, x, y, w, h))
        
        # 各フレームに対して bbox 描画
        id_color = {}  # 各 track_id ごとにランダムな色を割り当て
        frame_idx = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in bbox_data:
                for (track_id, x, y, w, h) in bbox_data[frame_idx]:
                    if track_id not in id_color:
                        id_color[track_id] = (
                            random.randint(0, 255),
                            random.randint(0, 255),
                            random.randint(0, 255)
                        )
                    color = id_color[track_id]
                    
                    # bboxの座標計算とクランプ処理
                    x1 = x
                    y1 = y
                    x2 = x + w
                    y2 = y + h
                    if x2 < 0 or y2 < 0 or x1 > width or y1 > height:
                        continue
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # bboxの左上に赤文字でtrack_idを描画
                    text_x = x1
                    text_y = y1 - 5 if y1 - 5 > 10 else y1 + 15
                    cv2.putText(frame, str(track_id), (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            out.write(frame)
            frame_idx += 1
        cap.release()
        out.release()
        print(f"{video_file} のbbox描画動画を {output_video_path} に保存しました。")
        
print("全ての処理が完了しました。")
