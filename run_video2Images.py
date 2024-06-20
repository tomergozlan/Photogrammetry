import argparse
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
from tqdm import tqdm


def select_100_distinct_frames(video_path, output_folder, overlap_threshold=0.2):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Could not open video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames < 100:
        raise Exception("Video does not have enough frames.")

    # Calculate the number of frames to skip at the beginning and end
    frames_to_skip = int(0.04 * total_frames)

    # Create output directory if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize variables
    selected_frames_count = 0
    last_frame = None
    frame_indices = []

    with tqdm(total=total_frames - 2 * frames_to_skip, desc="Analyzing Frames", bar_format='{l_bar}{bar} | Elapsed Time: {elapsed}') as pbar:
        for frame_idx in range(frames_to_skip, total_frames - frames_to_skip):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if last_frame is not None:
                score = ssim(last_frame, gray_frame, data_range=gray_frame.max() - gray_frame.min())
                if score < (1 - overlap_threshold):
                    frame_indices.append(frame_idx)
                    last_frame = gray_frame
            else:
                # Always save the first frame after skipping
                frame_indices.append(frame_idx)
                last_frame = gray_frame

            pbar.update(1)

    if len(frame_indices) < 100:
        raise Exception("Not enough distinct frames found.")

    # Sample 100 frames from the filtered indices
    selected_indices = np.linspace(0, len(frame_indices) - 1, 100, dtype=int)
    selected_indices = [frame_indices[i] for i in selected_indices]

    with tqdm(total=100, desc="Selecting Frames") as pbar:
        for frame_idx in selected_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_filename = os.path.join(output_folder, f"frame_{selected_frames_count + 1:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            selected_frames_count += 1
            pbar.update(1)

    cap.release()
    if selected_frames_count < 100:
        print(f"Not enough distinct frames found. Only {selected_frames_count} frames were selected.")
    else:
        print(f"Selected frames saved to folder '{output_folder}'.")

def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video based on structural dissimilarity and removing first and last 2% of frames.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("output_folder", type=str, help="Directory to save the extracted frames.")
    parser.add_argument("--overlap_threshold", type=float, default=0.2, help="Overlap threshold for frame selection.")

    args = parser.parse_args()
    select_100_distinct_frames(args.video_path, args.output_folder, args.overlap_threshold)


if __name__ == "__main__":
    main()

    print("Done.")
