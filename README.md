# Photogrammetry Project Report (Part 2)

## Abstract

In this part of the photogrammetry project, we focus on two primary objectives:
1. Computing a set of images from video optimized for photogrammetry.
2. Implementing a software tool to stitch two sets of point clouds (pc1, pc2) into a single point cloud using the CloudCompare open-source Command Line Interface (CLI).

The video2Images tool is designed to process new video clips (v1, v2) to generate two sets (s1, s2) of 100 images each, facilitating the creation of PointCloud p1 from s1 and PointCloud p2 from s2. Additionally, we developed a CLI tool for combining these point clouds, accommodating overlaps. Detailed documentation and validation using new videos (v3, v4) are included to ensure the reliability and efficacy of the tools.

![Flight Plan](flight_plan.png)
*Figure 1: The original flight plan of the videos of V1, V2 (mostly @ 100 meter altitude - about 785 m above sea level, created by Mavic air 2s). Note: in the flight, we have added a few "unknown" detours to allow testing of the Video2Images tool.*

## Introduction to `video2images.py`

Photogrammetry has become an essential tool for creating accurate 3D models from 2D images. This tool automates the selection of distinct frames from video footage, specifically tailored for photogrammetry applications, using structural similarity metrics and height estimation to ensure the selected frames are optimized for detailed 3D model construction.

The primary goals include:
- Increasing the robustness of the 3D model.
- Optimizing workflow efficiency by systematically selecting structurally distinct frames captured at an appropriate height.
- Maintaining a 20% overlap between frames to provide sufficient redundancy and improve model robustness.

## Script Logic and Implementation

### Overview

The script extracts 100 distinct frames from a video, optimized for photogrammetry by ensuring minimal overlap and filtering based on the drone's height.

### Height Estimation

The `estimate_height_from_frame` function estimates the drone's height from a video frame using the apparent size of a reference object in the frame. This function helps avoid frames taken during the landing and takeoff phases of the drone, ensuring that only frames captured from the air are selected for further processing.

```python
def estimate_height_from_frame(frame, reference_height=1.0, reference_size=100):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 0:
            estimated_distance = (reference_size / np.sqrt(area)) * reference_height
            return estimated_distance
    return None
```

### Frame Selection Logic
The select_100_distinct_frames function handles the frame selection process. It opens the video, calculates the total frames, and initializes variables for tracking selected frames. The function then iterates through each frame, estimating the height and filtering out frames below the height threshold. Selected frames are sampled to ensure distinctiveness based on SSIM scores and are saved to the output folder.

Maintaining an overlap of 20% between frames is crucial for improving the robustness of the 3D model. This overlap provides sufficient redundancy, ensuring that enough common points are available for the photogrammetry software to accurately align and reconstruct the 3D structure. It also helps in mitigating errors that might arise from occasional poor-quality frames or rapid changes in the scene.

```python
def select_100_distinct_frames(video_path, output_folder, overlap_threshold=0.2, height_threshold=50):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Could not open video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames < 100:
        raise Exception("Video does not have enough frames.")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    selected_frames_count = 0
    last_frame = None
    frame_indices = []

    with tqdm(total=total_frames, desc="Filtering Takeoff/Landing and Analyzing Frames", 
              bar_format='{l_bar}{bar} | Elapsed Time: {elapsed}') as pbar:
        for frame_idx in range(total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            height = estimate_height_from_frame(frame)
            if height is not None and height >= height_threshold:
                frame_indices.append(frame_idx)
            pbar.update(1)

    if len(frame_indices) < 100:
        raise Exception("Not enough frames with sufficient height.")

    selected_indices = np.linspace(0, len(frame_indices) - 1, 100, dtype=int)
    selected_indices are [frame_indices[i] for i in selected_indices]

    with tqdm(total=100, desc="Selecting Frames") as pbar:
        for frame_idx in selected_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if last_frame is not None:
                score = ssim(last_frame, gray_frame, data_range=gray_frame.max() - gray_frame.min())
                if score < (1 - overlap_threshold):
                    frame_filename = os.path.join(output_folder, f"frame_{selected_frames_count + 1:04d}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    selected_frames_count += 1
                    last_frame = gray_frame
                    pbar.update(1)
            else:
                frame_filename = os.path.join(output_folder, f"frame_{selected_frames_count + 1:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                selected_frames_count += 1
                last_frame = gray_frame
                pbar.update(1)

            if selected_frames_count == 100:
                break

    cap.release()
    if selected_frames_count < 100:
        print(f"Not enough distinct frames found. Only {selected_frames_count} frames were selected.")
    else:
        print(f"Selected frames saved to folder '{output_folder}'.")
```

### Command Line Interface (CLI)
The main function sets up a CLI using the argparse library. It allows users to specify the video path, output folder, overlap threshold, and height threshold. The CLI calls the select_100_distinct_frames function with the provided arguments, enabling flexible and user-friendly execution.

```python

def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video based on structural dissimilarity and height threshold.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("output_folder", type=str, help="Directory to save the extracted frames.")
    parser.add_argument("--overlap_threshold", type=float, default=0.2, help="Overlap threshold for frame selection.")
    parser.add_argument("--height_threshold", type=float, default=50, help="Height threshold to filter frames.")

    args = parser.parse_args()
    select_100_distinct_frames(args.video_path, args.output_folder, args.overlap_threshold, args.height_threshold)

if __name__ == "__main__":
    main()
    print("Done.")

```
Usage

The CLI for the script can be used as follows:

```sh
python video2images.py video_path output_folder [--overlap_threshold OVERLAP_THRESHOLD] [--height_threshold HEIGHT_THRESHOLD]
```
### Example

Here is an example of how to run the script using the CLI:
```sh
python select_frames.py input_video.mp4 output_frames --overlap_threshold 0.2 --height_threshold 50
```
This command processes the input_video.mp4 file, saves the selected frames to the output_frames directory, and uses an overlap threshold of 20% and a height threshold of 50 meters.


