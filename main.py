import cv2
from skimage.metrics import structural_similarity as ssim
import os
import gc
from tqdm import tqdm
import numpy as np

"""
    Optimized for photogrammetry, this function analyzes a video file to extract frames that differ significantly
    from each other based on the Structural Similarity Index (SSIM). It's designed to reduce redundancy in frame
    selection, which is crucial for efficient and effective 3D modeling.

    Args:
    video_path (str): Path to the video file.
    similarity_threshold (float): SSIM threshold below which a frame is considered dissimilar enough to be saved.
    output_folder (str): Directory where selected frames will be saved. Created if not existing.
    scale_factor (float): Factor by which to scale down frames to balance processing speed and detail.

    Returns:
    list: A list of selected frames that are below the SSIM threshold, suitable for photogrammetry software input.

    Photogrammetry requires a series of overlapping (15%-20%) images of a scene taken from different viewpoints to construct a 3D model.
    However, too many similar frames can increase processing time and computational costs without adding much value. This function
    selectively saves frames that provide new information, ensuring efficient data processing with sufficient overlap. Each frame
    is downscaled to reduce computational load, converted to grayscale for SSIM comparison, and compared to the previous frame.
    Frames that are sufficiently different (below the specified SSIM threshold) are saved, reducing the number of images to process
    for 3D reconstruction while maintaining essential overlap and coverage.
    """

def select_similar_frames(video_path, similarity_threshold=0.8, output_folder='selected_frame_similar_frames', scale_factor=0.5):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Could not open video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames
    selected_frames = []
    last_frame = None
    frame_index = 0

    # Create directory if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize the progress bar
    with tqdm(total=total_frames, desc="Analyzing Frames", unit='frame') as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to reduce resolution and memory usage
            frame = cv2.resize(frame, (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor)))

            # Convert frame to grayscale for SSIM computation
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)  # Using float32 to save memory

            if last_frame is None:
                # Always select the first frame
                selected_frames.append(frame)
                cv2.imwrite(f'{output_folder}/frame_{frame_index:04d}.jpg', frame)
                last_frame = gray_frame
            else:
                # Calculate SSIM with the last selected frame
                score = ssim(last_frame, gray_frame, data_range=255)  # specify data_range for float images

                if score < similarity_threshold:
                    # If the new frame is different enough, select it
                    selected_frames.append(frame)
                    cv2.imwrite(f'{output_folder}/frame_{frame_index:04d}.jpg', frame)
                    last_frame = gray_frame

            frame_index += 1
            pbar.update(1)  # Update the progress bar

            # Periodically invoke garbage collection
            if frame_index % 100 == 0:
                gc.collect()

    cap.release()
    return selected_frames

"""
This function selects 100 frames from a video by dividing it into evenly spaced segments and identifying
the frame with the lowest SSIM score in each segment, indicating significant visual change.
It ensures frames are uniformly distributed and represent key changes, ideal for detailed 3D photogrammetry.
The process includes two passes: first to calculate SSIM scores and second to select the lowest SSIM frame per segment.
An exception is raised if the video has fewer than 100 frames, ensuring the algorithm only proceeds with adequate data.
"""

def select_100_frames(video_path, output_folder='selected_frames_lowest SSIM', scale_factor=0.5, frame_count=100):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Could not open video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames in the video
    if total_frames < frame_count:
        raise Exception("Not enough frames in the video to select 100 distinct frames.")

    # Calculate interval for frame selection
    interval = total_frames // frame_count

    selected_frames = []
    frame_scores = []
    last_frame = None
    frame_index = 0

    # Create directory if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # First pass: collect SSIM scores to determine the best frames in each segment
    with tqdm(total=total_frames, desc="Analyzing Video", unit='frame') as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize and convert frame for processing
            frame = cv2.resize(frame, (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor)))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

            if last_frame is not None:
                score = ssim(last_frame, gray_frame, data_range=gray_frame.max() - gray_frame.min())
                frame_scores.append((frame_index, score, frame))  # Store frame index, SSIM score, and frame

            last_frame = gray_frame
            frame_index += 1
            pbar.update(1)

            # Call garbage collector manually to free up memory
            if frame_index % 100 == 0:
                gc.collect()

    cap.release()

    # Second pass: select the best frame based on the lowest SSIM in each segment
    with tqdm(total=frame_count, desc="Selecting Frames", unit='frame') as pbar:
        for i in range(frame_count):
            segment_start = i * interval
            segment_end = (i + 1) * interval if i < frame_count - 1 else total_frames
            # Find the frame with the lowest SSIM score in the current segment
            best_frame = min(frame_scores[segment_start:segment_end], key=lambda x: x[1], default=None)
            if best_frame:
                selected_frames.append(best_frame[2])
                frame_path = os.path.join(output_folder, f'frame_{best_frame[0]:04d}.jpg')
                cv2.imwrite(frame_path, best_frame[2])
                pbar.update(1)

                # Call garbage collector after saving each frame
                gc.collect()

    return selected_frames

"""
    Analyzes a video to select frames based on structural similarity index (SSIM), then picks 100 frames 
    evenly spaced from the subset of dissimilar frames for varied representation.

    This function first filters out frames that are significantly different from each preceding frame based
    on the SSIM score. It then selects 100 frames from these dissimilar frames at regular intervals, ensuring
    that the frames are evenly distributed throughout the duration of the video. This method is particularly
    useful for scenarios requiring a representative yet manageable subset of a video for detailed analysis or
    presentation.

    Parameters:
        video_path (str): Path to the video file.
        similarity_threshold (float): Threshold for SSIM below which frames are considered dissimilar.
        output_folder (str): Directory where selected frames will be saved. Created if not existing.
        scale_factor (float): Factor by which the frame dimensions are reduced for SSIM computation.

    Returns:
        list: A list of numpy arrays representing the selected frames.

    Raises:
        Exception: If the video cannot be opened or if there are not enough dissimilar frames to select 100 frames.

    """

def select_100_similar_frames(video_path, similarity_threshold=0.8, output_folder='selected_frame_100_similar_frames', scale_factor=0.5):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Could not open video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames
    selected_frames = []
    last_frame = None
    frame_index = 0
    frame_details = []

    # Create directory if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize the progress bar
    with tqdm(total=total_frames, desc="Analyzing Frames", unit='frame') as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to reduce resolution and memory usage
            frame = cv2.resize(frame, (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor)))

            # Convert frame to grayscale for SSIM computation
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)  # Using float32 to save memory

            if last_frame is None:
                # Always select the first frame
                frame_details.append((frame_index, frame))
            else:
                # Calculate SSIM with the last selected frame
                score = ssim(last_frame, gray_frame, data_range=255)  # specify data_range for float images

                if score < similarity_threshold:
                    # If the new frame is different enough, select it
                    frame_details.append((frame_index, frame))

            last_frame = gray_frame
            frame_index += 1
            pbar.update(1)  # Update the progress bar

            # Periodically invoke garbage collection
            if frame_index % 100 == 0:
                gc.collect()

    cap.release()

    # Select 100 frames evenly from the list of selected frames based on dissimilarity
    if len(frame_details) < 100:
        raise Exception("Not enough diverse frames were selected to choose 100 frames.")
    interval = len(frame_details) // 100

    final_selected_frames = []
    for i in range(100):
        frame_idx, frame = frame_details[i * interval]
        frame_filename = f"{output_folder}/frame_{frame_idx + 1:04d}.jpg"  # Frame index starts at 1 for filenames
        cv2.imwrite(frame_filename, frame)
        final_selected_frames.append(frame)

    return final_selected_frames

"""
    Selects 100 evenly spaced frames from a video. This method ensures that the selected frames are
    distributed throughout the entire video, which is useful for creating evenly sampled snapshots
    for analysis or presentation.

    Parameters:
        video_path (str): The path to the video file.
        output_folder (str): The directory where the selected frames will be saved. Created if not existing.

    Returns:
        list: A list of numpy arrays representing the selected frames.
    """
def select_100_frames_interval_based(video_path, output_folder='selected_frames_interval_based'):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Could not open video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // 100  # Compute the interval for frame selection

    # Create directory if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    selected_frames = []
    frame_indices = [i * interval for i in range(100)]  # Calculate frame indices to capture

    # Initialize the progress bar
    with tqdm(total=100, desc="Selecting Frames", unit='frame') as pbar:
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # Move to the frame index
            ret, frame = cap.read()  # Read the frame
            if ret:
                frame_filename = f"{output_folder}/frame_{frame_index + 1:04d}.jpg"
                cv2.imwrite(frame_filename, frame)  # Save the frame
                selected_frames.append(frame)  # Add the frame to the list
                pbar.update(1)  # Update the progress bar

    cap.release()
    return selected_frames

# Usage
video_path = 'C:\\Users\\Tomer\\Desktop\\Photogrammetry\\data\\v1.mp4'

frames1 = select_similar_frames(video_path)
print(f"Selected {len(frames1)} frames and saved to folder 'selected_frame'.")

frames2 = select_100_frames(video_path)
print(f"Selected {len(frames2)} frames and saved to folder 'selected_frame'.")

frames3 = select_100_similar_frames(video_path)
print(f"Selected {len(frames3)} frames and saved to folder 'selected_frame'.")

frames4 = select_100_frames_interval_based(video_path)
print(f"Selected {len(frames4)} frames and saved to folder 'selected_frame'.")
