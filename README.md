# Photogrammetry Project Report (Part 2)

## Abstract

In this part of the photogrammetry project, we focus on two primary objectives:
1. Computing a set of images from video optimized for photogrammetry.
2. Implementing a software tool to stitch two sets of point clouds (pc1, pc2) into a single point cloud using the CloudCompare open-source Command Line Interface (CLI).

The video2Images tool is designed to process new video clips (v1, v2) to generate two sets (s1, s2) of 100 images each, facilitating the creation of PointCloud p1 from s1 and PointCloud p2 from s2. Additionally, we developed a CLI tool for combining these point clouds, accommodating overlaps. Detailed documentation and validation using new videos (v3, v4) are included to ensure the reliability and efficacy of the tools.

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

## Introduction to `images2model.py`

Meshroom is designed as a modular system where the heavy lifting is done by standalone command-line C++ programs. Meshroom itself is a lightweight Python wrapper that orchestrates these tasks. Instead of using the Meshroom GUI, we will directly call these command-line programs. The source code is open, enabling the possibility of directly linking to the libraries if needed.

### Requirements
- Meshroom/AliceVision
- Python (if not already installed)
- MeshLab (Optional) - for viewing PLY point files

The Python script `run_images2model.py` requires five arguments to run:

- `baseDir`: This is the directory where intermediary files will be stored.
- `imgDir`: This is the directory containing your source images.
- `binDir`: This is the directory containing the AliceVision executable files.
- `numImages`: This represents the number of images in the `imgDir`. For our example, it is 100.
- `runStep`: This specifies the operation to execute.

The command to run the script follows this format:

```bash
python run_images2model.py <baseDir> <imgDir> <binDir> <numImages> <runStep>
```

With the run_images2model.py python script, we are going to create this directory structure:

Directory Structure

Each directory corresponds to a specific step in the process. You can execute these steps individually by running the run_images2model_runXX.bat files. Alternatively, you can run all the steps sequentially using the run_images2model_all.bat file, optimizing the ability to understand the workflow required to prepare a 3D model.

## Workflow Steps

### 00_CameraInit
The initial step generates an SFM file. These SFM files are JSON files containing camera sizes, sensor information, detected 3D points (observations), distortion coefficients, and other related data. The initial SFM file in this directory will include only sensor information, with defaults chosen from a local sensor database. Later stages will generate SFM files that encompass complete camera extrinsic matrices, bundled points, and more.

### 01_FeatureExtraction
In this step, features and their descriptors are extracted from the images. The file extension will change depending on the type of feature being extracted.

### 02_ImageMatching
This preprocessing step identifies which images should be matched. For a set of 1000 images, checking each image against every other image would result in 1 million pairs, which is impractical. The 02_ImageMatching step reduces the number of pairs to be checked.

### 03_FeatureMatching
This step finds correspondences between images using the extracted feature descriptors. The output is a series of text files that are self-explanatory.

### 04_StructureFromMotion
This major step solves for the camera positions and intrinsics based on the found correspondences. The term “Structure From Motion” (SFM) is used generically to describe the process of solving camera positions. In setups with multiple synchronized cameras, SFM aligns the cameras even if the scene itself is static.

By default, Meshroom saves the solved data as an Alembic file, but I prefer to save it as an SFM file. This step generates intermediary data, allowing you to verify the correct alignment of the cameras. The script outputs PLY files, which can be viewed in MeshLab. The key files are:

bundle.sfm: SFM file with all observations.
cameras.sfm: SFM file with only the aligned cameras.
cloud_and_poses.ply: Contains found points and camera positions.
Cloud and Poses
Figure 2: The cloud_and_poses.ply file is particularly useful. The green dots representing cameras in this file provide a straightforward way to verify correct camera alignment. If issues arise, you can adjust features, matches, or SFM parameters accordingly.

### 05_PrepareDenseScene
05_PrepareDenseScene’s primary function is to undistort the images. It generates undistorted EXR images so that the following depth calculation and projection steps do not have to convert back and forth from the distortion function.

### 06_CameraConnection
This step slightly diverges from the intended workflow, where each folder represents an independent standalone step. In 06_CameraConnection, the process creates the camsPairsMatrixFromSeeds.bin file within the 05_PrepareDenseScene directory. This is necessary because the file needs to reside in the same directory as the undistorted images.

### 07_DepthMap
Generating depth maps is the most time-consuming step in AliceVision. This step creates a depth map for each image, saved as an EXR file. To make it more visible, I adjusted the settings, which clearly show details like the buildings and the trees.
Due to the lengthy nature of this process, there is an option to run groups of cameras as separate commands. For example, with 1000 cameras, you can process the depth maps in groups across multiple machines in a render farm. Alternatively, processing in smaller groups ensures that if one machine crashes, you don't need to rerun the entire process.

### 08_DepthMapFilter
The initial depth maps generated may contain inconsistencies, with some maps indicating visibility of areas that are occluded in others. The 08_DepthMapFilter step addresses this issue by identifying and isolating these inconsistent regions, ensuring depth consistency across all depth maps.

### 09_Meshing
This step is where the actual mesh generation begins. The 09_Meshing step creates the 3D mesh from the filtered depth maps. There may be issues with the initial mesh, but these can be resolved with subsequent refinement and processing steps.

### 10_MeshFiltering
The 10_MeshFiltering step refines the mesh generated in 09_Meshing. This step includes:

- Smoothing the mesh.
- Removing large, unnecessary triangles.
- Retaining the largest contiguous mesh while removing all smaller, disconnected parts.
- Mesh Filtering

### 11_Texturing
The final step, 11_Texturing, involves creating UV maps and projecting textures onto the mesh. With the completion of this step, the process is finished!
