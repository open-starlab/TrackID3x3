import os
import cv2
from collections import defaultdict

txt_dir = "../../output/CAMELTrack_outputs/Outdoor/filtered_MOT/"
video_dir = "../../videos/Outdoor/top/split/"
output_dir = "../../jersey-number-pipeline/data/Outdoor/test/images/"
os.makedirs(output_dir, exist_ok=True)

def load_mot(txt_file):
    """Reads MOT format track data"""
    track_data = defaultdict(list)
    with open(txt_file, "r") as f:
        for line in f:
            frame_id, track_id, x, y, w, h, _, _, _, _ = map(float, line.strip().split(","))
            track_data[int(frame_id)].append((int(track_id), int(x), int(y), int(w), int(h)))
    return track_data

for video_file in os.listdir(video_dir):
    if not video_file.endswith(".MOV"):
        continue

    video_name = os.path.splitext(video_file)[0]
    mot_file = os.path.join(txt_dir, f"{video_name}.txt")
    
    if not os.path.exists(mot_file):
        print(f"Skipping {video_file}, no corresponding MOT file.")
        continue
    
    # Loading MOT data
    track_data = load_mot(mot_file)
    
    # Create an output folder for each video
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    video_path = os.path.join(video_dir, video_file)
    cap = cv2.VideoCapture(video_path)
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        original_frame = frame.copy()  # Original image before cropping
        frame_h, frame_w = original_frame.shape[:2]
        
        # If there is track data for the current frame, each bbox area is cut out and saved
        if frame_idx in track_data:
            for track_id, x, y, w, h in track_data[frame_idx]:
                # Clip bbox range to image size
                x1 = max(x, 0)
                y1 = max(y, 0)
                x2 = min(x + w, frame_w)
                y2 = min(y + h, frame_h)
                
                # If range is invalid, skip
                if x1 >= x2 or y1 >= y2:
                    print(f"Invalid crop region for frame {frame_idx} track {track_id}: {(x1, y1, x2, y2)}")
                    continue
                
                cropped_roi = original_frame[y1:y2, x1:x2]
                
                # Create folders for each track
                track_dir = os.path.join(video_output_dir, f"track{track_id}")
                os.makedirs(track_dir, exist_ok=True)
                
                output_filename = f"{video_name}_frame{frame_idx}_track{track_id}_cropped.png"
                output_path = os.path.join(track_dir, output_filename)
                success = cv2.imwrite(output_path, cropped_roi)
                if not success:
                    print(f"Failed to save image for frame {frame_idx} track {track_id}")
        
        frame_idx += 1
    
    cap.release()
    print(f"Processed {video_file}")
print("Done.")
