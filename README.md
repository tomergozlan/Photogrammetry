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
```python

### Frame Selection Logic
The select_100_distinct_frames function handles the frame selection process. It opens the video, calculates the total frames, and initializes variables for tracking selected frames. The function then iterates through each frame, estimating the height and filtering out frames below the height threshold. Selected frames are sampled to ensure distinctiveness based on SSIM scores and are saved to the output folder.

Maintaining an overlap of 20% between frames is crucial for improving the robustness of the 3D model. This overlap provides sufficient redundancy, ensuring that enough common points are available for the photogrammetry software to accurately align and reconstruct the 3D structure. It also helps in mitigating errors that might arise from occasional poor-quality frames or rapid changes in the scene.
