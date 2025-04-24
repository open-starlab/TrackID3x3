import os
import glob
import re
import json
import subprocess

# --- Setting items ---
input_dir = "../../videos/Outdoor/top"          # Folder with the original video files
csv_dir = "../../ground_truth/Outdoor/delimitation_frames"  # Folder with CSV files
output_dir = "../../videos/Outdoor/top/split"  # Video output destination after splitting
os.makedirs(output_dir, exist_ok=True) 

# Specify the target video file name (excluding the file extension)
target_files = {"IMG_0106"}

def get_frame_pts_mapping(video_path):
    """
    Use ffprobe to get the PTS (best_effort_timestamp_time) for all frames in the video,
    Returns a dictionary whose keys are the frame numbers starting from 0 and whose values are the PTS (in seconds).
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "json",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    data = json.loads(result.stdout)
    
    pts_mapping = {}
    frames = data.get("frames", [])
    for i, frame in enumerate(frames):
        if "best_effort_timestamp_time" in frame:
            pts = float(frame["best_effort_timestamp_time"])
            pts_mapping[i] = pts
        else:
            pts_mapping[i] = None
    return pts_mapping

def extract_frames_from_csv(csv_file):
    """
    Obtains only the first column of each row from the CSV file,
    extracts the frame number at the beginning of that column and returns it as a list.
    If the output of BoT-SORT starts with 1, it is converted to start with 0 for internal processing (-1 is applied).
    """
    frames = []
    with open(csv_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Get the first column separated by commas
            first_col = line.split(",")[0]
            match = re.match(r"(\d+)", first_col)
            if match:
                frame_num = int(match.group(1)) - 1  # Converted from beginning 1 to beginning 0
                frames.append(frame_num)
    return frames

# Get all MOV files in input_dir
mov_files = glob.glob(os.path.join(input_dir, "*.MOV"))

for mov_file in mov_files:
    base_name = os.path.splitext(os.path.basename(mov_file))[0]  # Example: “IMG_0106”
    if base_name not in target_files:
        continue  # Skip if not eligible
    
    csv_file = os.path.join(csv_dir, base_name + ".csv")
    
    if not os.path.isfile(csv_file):
        print(f"[WARN] Skip due to missing CSV: {csv_file}")
        continue

    # Get frame number (converted to start with 0) from CSV
    frame_numbers = extract_frames_from_csv(csv_file)
    if len(frame_numbers) < 2:
        print(f"[WARN] Skip due to less than two frame numbers: {csv_file}")
        continue
    if len(frame_numbers) % 2 != 0:
        print(f"[WARN] Odd frame numbers (not divisible by pairs): {csv_file}")
        continue

    # Obtain mapping between frame number and PTS of video
    pts_mapping = get_frame_pts_mapping(mov_file)
    total_frames = len(pts_mapping)
    print(f"[INFO] {base_name}: Total frames = {total_frames}")

    print(f"\n=== {base_name} Start the division of ===")
    segment_index = 1
    # Processed in pairs (odd to even rows)
    for i in range(0, len(frame_numbers), 2):
        start_frame = frame_numbers[i]
        end_frame   = frame_numbers[i+1]

        # Get PTS from mapping
        if start_frame not in pts_mapping or end_frame not in pts_mapping:
            print(f"[WARN] Frame number {start_frame} or {end_frame} does not exist in the PTS mapping.")
            continue

        start_time = pts_mapping[start_frame]
        end_time   = pts_mapping[end_frame]
        if start_time is None or end_time is None:
            print(f"[WARN] Unable to obtain PTS for frame {start_frame} or {end_frame}.")
            continue

        output_file = os.path.join(output_dir, f"{base_name}_{segment_index}.MOV")
        
        # Extract based on t (seconds) with ffmpeg's select filter
        command = [
            "ffmpeg", "-y",
            "-i", mov_file,
            "-vf", f"select='between(t,{start_time},{end_time})',setpts=PTS-STARTPTS",
            "-af", f"aselect='between(t,{start_time},{end_time})',asetpts=PTS-STARTPTS",
            "-c:v", "libx264",
            "-crf", "18",            # CRF value
            "-preset", "veryfast",
            "-c:a", "aac",           # Changed audio from ALAC to AAC
            "-b:a", "128k",          # Encode at 128 kbps
            output_file
        ]

        # When outputting, the original BoT-SORT number (starting from 1) is displayed.
        print(f"[RUN] {output_file} → フレーム {start_frame+1} ~ {end_frame+1}  (PTS {start_time:.3f} ~ {end_time:.3f}秒)")
        subprocess.run(command, check=True)
        segment_index += 1

print("Finished.")
