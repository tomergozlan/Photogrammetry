# Photogrammetry Project

This project, conducted at Ariel University, School of Computer Science, focuses on the algorithms and implementation challenges of performing photogrammetry from a video. The project is divided into three main parts, each with specific objectives and tasks. The end goal is to create accurate 3D models from video footage.

## Table of Contents
- [General Information](#general-information)
- [Data](#data)
- [Tasks](#tasks)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Project](#running-the-project)
  - [Video to Images](#video-to-images)
  - [Images to Model](#images-to-model)
  - [Merging Point Clouds](#merging-point-clouds)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## General Information
This project is part of the undergraduate curriculum in the Computer Science department at Ariel University. It aims to develop a comprehensive understanding of photogrammetry by implementing a system that converts video footage into 3D models.

## Data
The project uses the following videos:
- [Video 1](https://youtu.be/DszOxc3r-WM)
- [Video 2](https://www.youtube.com/watch?v=Q-O_Y_3ypn4)
- [Video 3](https://youtu.be/LXzSFUHa5mM)

## Tasks
### Part 1
1. Research and summarize the field of photogrammetry.
2. Find, build, and understand the best open-source photogrammetry tool.
3. Perform an initial performance evaluation using the chosen tool and provided video.
4. Write a project report using Overleaf.

### Part 2
1. Extract a set of images from video optimized for photogrammetry.
2. Implement a software tool to stitch two sets of point clouds into a single point cloud using CloudCompare CLI.
3. Validate the tools with new videos.

### Part 3
1. Create and publish photogrammetry missions for the University region.
2. Merge additional photogrammetry data from ground videos.
3. Summarize the results in a project report.

## Prerequisites
Before you begin, ensure you have the following tools installed:
- Python
- Meshroom/AliceVision
- MeshLab (Optional, for viewing PLY point files)
- CloudCompare

## Installation
1. Clone the project repository:
   ```bash
   git clone [repository link]
   cd [repository folder]
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
 
## Running the Project
### Video to Images
Extract frames from video using the run_video2images.py script:
```bash
python run_video2images.py video_path output_folder --overlap_threshold 0.2 --height_threshold 50

### Images to Model
Generate 3D models from images using the run_images2model.py script:
```bash
python run_images2model.py <baseDir> <imgDir> <binDir> <numImages> <runStep>

### Merging Point Clouds
Merge point clouds using the run_merge_clouds.py script:
```bash
python run_merge_clouds.py <point_cloud_1> <point_cloud_2> <output_path> [--color_pc1 <R,G,B>] [--color_pc2 <R,G,B>]

## Contributing
If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Make your changes and commit them (git commit -m 'Add some feature').
4. Push to the branch (git push origin feature-branch).
5. Open a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For any inquiries or further information, please contact:
* Batel Yerushalmi
* Tomer Gozlan
* Ashwaq Matar