import os
import cv2
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from collections import defaultdict

# path setting
txt_dir = "../../BoT-SORT_outputs/Indoor/filtered"
video_dir = "../../videos/Indoor"
output_dir = "../../segmentation_results/Indoor"
hist_output_dir = "../../color_histograms/Indoor"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(hist_output_dir, exist_ok=True)

# Setup of Detectron2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80  # COCOのクラス数
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
predictor = DefaultPredictor(cfg)

# Function to read MOT files
def load_mot(txt_file):
    """Reads MOT format track data"""
    track_data = defaultdict(list)
    with open(txt_file, "r") as f:
        for line in f:
            frame_id, track_id, x, y, w, h, _, _, _, _ = map(float, line.strip().split(","))
            track_data[int(frame_id)].append((int(track_id), int(x), int(y), int(w), int(h)))
    return track_data

# Video Processing
for video_file in os.listdir(video_dir):
    if not video_file.endswith(".mp4"):
        continue
    
    video_name = os.path.splitext(video_file)[0]
    mot_file = os.path.join(txt_dir, f"{video_name}.txt")
    
    if not os.path.exists(mot_file):
        print(f"Skipping {video_file}, no corresponding MOT file.")
        continue
    
    # Read MOT data
    track_data = load_mot(mot_file)
    
    # 各動画ごとの出力フォルダを作成
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    video_hist_dir = os.path.join(hist_output_dir, video_name)
    os.makedirs(video_hist_dir, exist_ok=True)
    
    video_path = os.path.join(video_dir, video_file)
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    output_path = os.path.join(video_output_dir, f"seg_{video_file}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    track_histograms = defaultdict(list)  # Save color histograms for each track
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        original_frame = frame.copy()  # 変更前の元のフレームを保存
        
        if frame_idx in track_data:
            outputs = predictor(frame)
            instances = outputs["instances"]
            pred_boxes = instances.pred_boxes.tensor.cpu().numpy() if instances.has("pred_boxes") else None
            pred_masks = instances.pred_masks.cpu().numpy() if instances.has("pred_masks") else None
            pred_classes = instances.pred_classes.cpu().numpy() if instances.has("pred_classes") else None
            
            for track_id, x, y, w, h in track_data[frame_idx]:
                if pred_boxes is not None and pred_masks is not None:
                    max_area = 0
                    best_mask = None
                    best_mask_index = -1
                    
                    # bbox領域内で最も面積が大きいマスクを探索
                    for i, box in enumerate(pred_boxes):
                        if pred_classes[i] != 0:  # クラス 0 は "person"
                            continue
                        
                        x1, y1, x2, y2 = map(int, box)
                        if (x1 < x + w and x2 > x and y1 < y + h and y2 > y):
                            mask = pred_masks[i].astype(np.uint8) * 255
                            # bbox領域に合わせてマスクを切り出す
                            mask = mask[y:y+h, x:x+w]
                            area = np.sum(mask > 0)
                            if area > max_area:
                                max_area = area
                                best_mask = mask
                                best_mask_index = i
                    
                    if best_mask is not None:
                        # 対象bbox内の領域を切り出す
                        cropped_roi = original_frame[y:y+h, x:x+w]
                        # クロップした領域に対してマスクを適用（非マスク部分を黒に）
                        cropped_masked_area = cv2.bitwise_and(cropped_roi, cropped_roi, mask=best_mask)
                        
                        # 1つの画像として保存（例：IMG_0104_1_frame0_track1_cropped_masked_area.png）
                        # cv2.imwrite(os.path.join(video_output_dir, f"{video_name}_frame{frame_idx}_track{track_id}_cropped_masked_area.png"), cropped_masked_area)
                        
                        # ヒストグラムも対象bbox内で計算
                        hist = cv2.calcHist([cropped_roi], [0, 1, 2], best_mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                        hist = cv2.normalize(hist, hist).flatten()
                        track_histograms[track_id].append(hist)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    print(f"Processed {video_file}")
    
    # 動画ごとのヒストグラム出力フォルダへ保存
    for track_id, hist_list in track_histograms.items():
        hist_med = np.median(hist_list, axis=0)
        np.save(os.path.join(video_hist_dir, f"{video_name}_track{track_id}.npy"), hist_med)

print("Segmentation and histogram extraction completed!")
