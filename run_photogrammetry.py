import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import pandas as pd
import subprocess
from sklearn.decomposition import PCA
import webbrowser
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

console = Console()


def display_main_menu():
    os.system('cls' if os.name == 'nt' else 'clear')

    header = Text(r"""
*****************************************************************************************************************************
*  ____  _           _                                                 _                   ____            _           _    *
* |  _ \| |__   ___ | |_ ___   __ _ _ __ __ _ _ __ ___  _ __ ___   ___| |_ _ __ _   _     |  _ \ _ __ ___ (_) ___  ___| |_  *
* | |_) | '_ \ / _ \| __/ _ \ / _` | '__/ _` | '_ ` _ \| '_ ` _ \ / _ \ __| '__| | | |    | |_) | '__/ _ \| |/ _ \/ __| __| *
* |  __/| | | | (_) | || (_) | (_| | | | (_| | | | | | | | | | | |  __/ |_| |  | |_| |    |  __/| | | (_) | |  __/ (__| |_  *
* |_|   |_| |_|\___/ \__\___/ \__, |_|  \__,_|_| |_| |_|_| |_| |_|\___|\__|_|   \__, |    |_|   |_|  \___// |\___|\___|\__| *
*                             |___/                                             |___/                   |__/                *
*****************************************************************************************************************************
    """, style="bold cyan")

    console.print(header)
    console.print(Panel(
        "[bold cyan]Final Project on Photogrammetry[/bold cyan]\n[bold cyan]Presenters: Tomer Gozlan, Batel Yerushalmi, Ashwaq Matar[/bold cyan]\n\nWelcome to the Photogrammetry Project!\nThis project focuses on the algorithms and implementation challenges of performing photogrammetry from a video.\nIt is part of an undergraduate course at Ariel University, School of Computer Science.",
        title="Project Information", title_align="left"))

    table = Table(title="\nMain Menu")
    table.add_column("Option", justify="center", style="cyan", no_wrap=True)
    table.add_column("Description", justify="left", style="magenta")

    table.add_row("1", "CODE")
    table.add_row("2", "SLIDERS")
    table.add_row("3", "MENU")
    table.add_row("4", "PROJECT REPORT (OVERLEAF)")
    table.add_row("5", "RESEARCH SURVEY (OVERLEAF)")

    console.print(table)
    console.print("\nPlease choose an option (1-5): ", end='')


def open_code():
    webbrowser.open("https://github.com/tomergozlan/Photogrammetry")
    console.print("Opening project code on GitHub...", style="bold green")


def open_sliders():
    webbrowser.open(
        "https://www.canva.com/design/DAGI2QZSjUc/AvDDLck72I9KDG_Cm2Rqjw/edit?utm_content=DAGI2QZSjUc&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton")  # Replace with actual path
    console.print("Opening sliders on Canva...", style="bold green")


def display_functionality_menu():
    table = Table(title="\nFunctionality Menu")
    table.add_column("Option", justify="center", style="cyan", no_wrap=True)
    table.add_column("Description", justify="left", style="magenta")

    table.add_row("1", "Produce frames from a certain video")
    table.add_row("2", "Produce a model from frames")
    table.add_row("3", "Create a merged point cloud")
    table.add_row("4", "Open a point cloud in CloudCompare")
    table.add_row("5", "Go back to the main menu")
    table.add_row("6", "Exit")

    console.print(table)
    console.print("\nPlease choose an option (1-6): ", end='')

def select_100_distinct_frames(video_path, output_folder, overlap_threshold=0.2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Could not open video.")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total_frames < 100:
        raise Exception("Video does not have enough frames.")
    frames_to_skip = int(0.04 * total_frames)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
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
                frame_indices.append(frame_idx)
                last_frame = gray_frame
            pbar.update(1)
    if len(frame_indices) < 100:
        raise Exception("Not enough distinct frames found.")
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

def SilentMkdir(theDir):
    try:
        os.mkdir(theDir)
    except:
        pass
    return 0

def Run_00_CameraInit(baseDir, binDir, srcImageDir):
    print("\033[95mRunning CameraInit...\033[0m")
    SilentMkdir(baseDir + "/00_CameraInit")
    binName = binDir + "/aliceVision_cameraInit.exe"
    dstDir = baseDir + "/00_CameraInit/"
    cmdLine = binName
    cmdLine = cmdLine + " --defaultFieldOfView 45.0 --verboseLevel info --sensorDatabase \"\" --allowSingleView 1"
    cmdLine = cmdLine + " --imageFolder \"" + srcImageDir + "\""
    cmdLine = cmdLine + " --output \"" + dstDir + "cameraInit.sfm\""
    print(cmdLine)
    os.system(cmdLine)
    return 0

def Run_01_FeatureExtraction(baseDir, binDir, numImages):
    print("\033[95mRunning FeatureExtraction...\033[0m")
    SilentMkdir(baseDir + "/01_FeatureExtraction")
    srcSfm = baseDir + "/00_CameraInit/cameraInit.sfm"
    binName = binDir + "/aliceVision_featureExtraction.exe"
    dstDir = baseDir + "/01_FeatureExtraction/"
    cmdLine = binName
    cmdLine = cmdLine + " --describerTypes sift --forceCpuExtraction True --verboseLevel info --describerPreset normal"
    cmdLine = cmdLine + " --rangeStart 0 --rangeSize " + str(numImages)
    cmdLine = cmdLine + " --input \"" + srcSfm + "\""
    cmdLine = cmdLine + " --output \"" + dstDir + "\""
    print(cmdLine)
    os.system(cmdLine)
    return 0

def Run_02_ImageMatching(baseDir, binDir):
    print("\033[95mRunning ImageMatching...\033[0m")
    SilentMkdir(baseDir + "/02_ImageMatching")
    srcSfm = baseDir + "/00_CameraInit/cameraInit.sfm"
    srcFeatures = baseDir + "/01_FeatureExtraction/"
    dstMatches = baseDir + "/02_ImageMatching/imageMatches.txt"
    binName = binDir + "/aliceVision_imageMatching.exe"
    cmdLine = binName
    cmdLine = cmdLine + " --minNbImages 200 --tree \"\" --maxDescriptors 500 --verboseLevel info --weights \"\" --nbMatches 50"
    cmdLine = cmdLine + " --input \"" + srcSfm + "\""
    cmdLine = cmdLine + " --featuresFolder \"" + srcFeatures + "\""
    cmdLine = cmdLine + " --output \"" + dstMatches + "\""
    print(cmdLine)
    os.system(cmdLine)
    return 0

def Run_03_FeatureMatching(baseDir, binDir):
    print("\033[95mRunning FeatureMatching...\033[0m")
    SilentMkdir(baseDir + "/03_FeatureMatching")
    srcSfm = baseDir + "/00_CameraInit/cameraInit.sfm"
    srcFeatures = baseDir + "/01_FeatureExtraction/"
    srcImageMatches = baseDir + "/02_ImageMatching/imageMatches.txt"
    dstMatches = baseDir + "/03_FeatureMatching"
    binName = binDir + "/aliceVision_featureMatching.exe"
    cmdLine = binName
    cmdLine = cmdLine + " --verboseLevel info --describerTypes sift --maxMatches 0 --exportDebugFiles False --savePutativeMatches False --guidedMatching False"
    cmdLine = cmdLine + " --geometricEstimator acransac --geometricFilterType fundamental_matrix --maxIteration 2048 --distanceRatio 0.8"
    cmdLine = cmdLine + " --photometricMatchingMethod ANN_L2"
    cmdLine = cmdLine + " --imagePairsList \"" + srcImageMatches + "\""
    cmdLine = cmdLine + " --input \"" + srcSfm + "\""
    cmdLine = cmdLine + " --featuresFolders \"" + srcFeatures + "\""
    cmdLine = cmdLine + " --output \"" + dstMatches + "\""
    print(cmdLine)
    os.system(cmdLine)
    return 0

def Run_04_StructureFromMotion(baseDir, binDir):
    print("\033[95mRunning StructureFromMotion...\033[0m")
    SilentMkdir(baseDir + "/04_StructureFromMotion")
    srcSfm = baseDir + "/00_CameraInit/cameraInit.sfm"
    srcFeatures = baseDir + "/01_FeatureExtraction/"
    srcImageMatches = baseDir + "/02_ImageMatching/imageMatches.txt"
    srcMatches = baseDir + "/03_FeatureMatching"
    dstDir = baseDir + "/04_StructureFromMotion"
    binName = binDir + "/aliceVision_incrementalSfm.exe"
    cmdLine = binName
    cmdLine = cmdLine + " --minAngleForLandmark 2.0 --minNumberOfObservationsForTriangulation 2 --maxAngleInitialPair 40.0 --maxNumberOfMatches 0 --localizerEstimator acransac --describerTypes sift --lockScenePreviouslyReconstructed False --localBAGraphDistance 1"
    cmdLine = cmdLine + " --initialPairA \"\" --initialPairB \"\" --interFileExtension .ply --useLocalBA True"
    cmdLine = cmdLine + " --minInputTrackLength 2 --useOnlyMatchesFromInputFolder False --verboseLevel info --minAngleForTriangulation 3.0 --maxReprojectionError 4.0 --minAngleInitialPair 5.0"
    cmdLine = cmdLine + " --input \"" + srcSfm + "\""
    cmdLine = cmdLine + " --featuresFolders \"" + srcFeatures + "\""
    cmdLine = cmdLine + " --matchesFolders \"" + srcMatches + "\""
    cmdLine = cmdLine + " --outputViewsAndPoses \"" + dstDir + "/cameras.sfm\""
    cmdLine = cmdLine + " --extraInfoFolder \"" + dstDir + "\""
    cmdLine = cmdLine + " --output \"" + dstDir + "/bundle.sfm\""
    print(cmdLine)
    os.system(cmdLine)
    return 0

def Run_05_PrepareDenseScene(baseDir, binDir):
    print("\033[95mRunning PrepareDenseScene...\033[0m")
    SilentMkdir(baseDir + "/05_PrepareDenseScene")
    srcSfm = baseDir + "/04_StructureFromMotion/bundle.sfm"
    dstDir = baseDir + "/05_PrepareDenseScene"
    binName = binDir + "/aliceVision_prepareDenseScene.exe"
    cmdLine = binName
    cmdLine = cmdLine + " --verboseLevel info"
    cmdLine = cmdLine + " --input \"" + srcSfm + "\""
    cmdLine = cmdLine + " --output \"" + dstDir + "\""
    print(cmdLine)
    os.system(cmdLine)
    return 0

def Run_06_CameraConnection(baseDir, binDir):
    print("\033[95mRunning CameraConnection...\033[0m")
    SilentMkdir(baseDir + "/06_CameraConnection")
    srcIni = baseDir + "/05_PrepareDenseScene/mvs.ini"
    binName = binDir + "/aliceVision_cameraConnection.exe"
    cmdLine = binName
    cmdLine = cmdLine + " --verboseLevel info"
    cmdLine = cmdLine + " --ini \"" + srcIni + "\""
    print(cmdLine)
    os.system(cmdLine)
    return 0

def Run_07_DepthMap(baseDir, binDir, numImages, groupSize):
    print("\033[95mRunning DepthMap...\033[0m")
    SilentMkdir(baseDir + "/07_DepthMap")
    numGroups = (numImages + (groupSize - 1)) / groupSize
    srcIni = baseDir + "/05_PrepareDenseScene/mvs.ini"
    binName = binDir + "/aliceVision_depthMapEstimation.exe"
    dstDir = baseDir + "/07_DepthMap"
    cmdLine = binName
    cmdLine = cmdLine + " --sgmGammaC 5.5 --sgmWSH 4 --refineGammaP 8.0 --refineSigma 15 --refineNSamplesHalf 150 --sgmMaxTCams 10 --refineWSH 3 --downscale 2 --refineMaxTCams 6 --verboseLevel info --refineGammaC 15.5 --sgmGammaP 8.0"
    cmdLine = cmdLine + " --refineNiters 100 --refineNDepthsToRefine 31 --refineUseTcOrRcPixSize False"
    cmdLine = cmdLine + " --ini \"" + srcIni + "\""
    cmdLine = cmdLine + " --output \"" + dstDir + "\""
    for groupIter in range(int(numGroups)):
        groupStart = groupSize * groupIter
        groupSize = min(groupSize, numImages - groupStart)
        print("DepthMap Group %d/%d: %d, %d" % (groupIter, numGroups, groupStart, groupSize))
        cmd = cmdLine + (" --rangeStart %d --rangeSize %d" % (groupStart, groupSize))
        print(cmd)
        os.system(cmd)
    return 0

def Run_08_DepthMapFilter(baseDir, binDir):
    print("\033[95mRunning DepthMapFilter...\033[0m")
    SilentMkdir(baseDir + "/08_DepthMapFilter")
    binName = binDir + "/aliceVision_depthMapFiltering.exe"
    dstDir = baseDir + "/08_DepthMapFilter"
    srcIni = baseDir + "/05_PrepareDenseScene/mvs.ini"
    srcDepthDir = baseDir + "/07_DepthMap"
    cmdLine = binName
    cmdLine = cmdLine + " --minNumOfConsistensCamsWithLowSimilarity 4"
    cmdLine = cmdLine + " --minNumOfConsistensCams 3 --verboseLevel info --pixSizeBall 0"
    cmdLine = cmdLine + " --pixSizeBallWithLowSimilarity 0 --nNearestCams 10"
    cmdLine = cmdLine + " --ini \"" + srcIni + "\""
    cmdLine = cmdLine + " --output \"" + dstDir + "\""
    cmdLine = cmdLine + " --depthMapFolder \"" + srcDepthDir + "\""
    print(cmdLine)
    os.system(cmdLine)
    return 0

def Run_09_Meshing(baseDir, binDir):
    print("\033[95mRunning Meshing...\033[0m")
    SilentMkdir(baseDir + "/09_Meshing")
    binName = binDir + "/aliceVision_meshing.exe"
    srcIni = baseDir + "/05_PrepareDenseScene/mvs.ini"
    srcDepthFilterDir = baseDir + "/08_DepthMapFilter"
    srcDepthMapDir = baseDir + "/07_DepthMap"
    dstDir = baseDir + "/09_Meshing"
    cmdLine = binName
    cmdLine = cmdLine + " --simGaussianSizeInit 10.0 --maxInputPoints 50000000 --repartition multiResolution"
    cmdLine = cmdLine + " --simGaussianSize 10.0 --simFactor 15.0 --voteMarginFactor 4.0 --contributeMarginFactor 2.0 --minStep 2 --pixSizeMarginFinalCoef 4.0 --maxPoints 5000000 --maxPointsPerVoxel 1000000 --angleFactor 15.0 --partitioning singleBlock"
    cmdLine = cmdLine + " --minAngleThreshold 1.0 --pixSizeMarginInitCoef 2.0 --refineFuse True --verboseLevel info"
    cmdLine = cmdLine + " --ini \"" + srcIni + "\""
    cmdLine = cmdLine + " --depthMapFilterFolder \"" + srcDepthFilterDir + "\""
    cmdLine = cmdLine + " --depthMapFolder \"" + srcDepthMapDir + "\""
    cmdLine = cmdLine + " --output \"" + dstDir + "/mesh.obj\""
    print(cmdLine)
    os.system(cmdLine)
    return 0

def Run_10_MeshFiltering(baseDir, binDir):
    print("\033[95mRunning MeshFiltering...\033[0m")
    SilentMkdir(baseDir + "/10_MeshFiltering")
    binName = binDir + "/aliceVision_meshFiltering.exe"
    srcMesh = baseDir + "/09_Meshing/mesh.obj"
    dstMesh = baseDir + "/10_MeshFiltering/mesh.obj"
    cmdLine = binName
    cmdLine = cmdLine + " --verboseLevel info --removeLargeTrianglesFactor 60.0 --iterations 5 --keepLargestMeshOnly True"
    cmdLine = cmdLine + " --lambda 1.0"
    cmdLine = cmdLine + " --input \"" + srcMesh + "\""
    cmdLine = cmdLine + " --output \"" + dstMesh + "\""
    print(cmdLine)
    os.system(cmdLine)
    return 0

def Run_11_Texturing(baseDir, binDir):
    print("\033[95mRunning Texturing...\033[0m")
    SilentMkdir(baseDir + "/11_Texturing")
    binName = binDir + "/aliceVision_texturing.exe"
    srcMesh = baseDir + "/10_MeshFiltering/mesh.obj"
    srcRecon = baseDir + "/09_Meshing/denseReconstruction.bin"
    srcIni = baseDir + "/05_PrepareDenseScene/mvs.ini"
    dstDir = baseDir + "/11_Texturing"
    cmdLine = binName
    cmdLine = cmdLine + " --textureSide 8192"
    cmdLine = cmdLine + " --downscale 2 --verboseLevel info --padding 15"
    cmdLine = cmdLine + " --unwrapMethod Basic --outputTextureFileType png --flipNormals False --fillHoles False"
    cmdLine = cmdLine + " --inputDenseReconstruction \"" + srcRecon + "\""
    cmdLine = cmdLine + " --inputMesh \"" + srcMesh + "\""
    cmdLine = cmdLine + " --ini \"" + srcIni + "\""
    cmdLine = cmdLine + " --output \"" + dstDir + "\""
    print(cmdLine)
    os.system(cmdLine)
    return 0

def read_ply(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    header = []
    data = []
    in_header = True
    for line in lines:
        if in_header:
            header.append(line.strip())
            if line.strip() == "end_header":
                in_header = False
        else:
            data.append([float(val) if i < 3 else int(val) for i, val in enumerate(line.strip().split())])
    df = pd.DataFrame(data, columns=['x', 'y', 'z', 'red', 'green', 'blue'])
    return header, df

def write_ply(file_path, header, df):
    with open(file_path, 'w') as f:
        for line in header:
            f.write(f"{line}\n")
        df.to_csv(f, sep=' ', index=False, header=False, float_format='%.6f')

def color_point_cloud(df, color):
    df[['red', 'green', 'blue']] = color
    return df

def bounding_box_size(df):
    min_values = df[['x', 'y', 'z']].min()
    max_values = df[['x', 'y', 'z']].max()
    size = max_values - min_values
    return size

def scale_point_cloud(df, scale_factor):
    df[['x', 'y', 'z']] *= scale_factor
    return df

def translate_to_origin(df):
    centroid = df[['x', 'y', 'z']].mean()
    df[['x', 'y', 'z']] -= centroid
    return df

def normalize_point_cloud(df):
    pca = PCA(n_components=3)
    points = df[['x', 'y', 'z']].values
    pca.fit(points)
    normalized_points = pca.transform(points)
    df[['x', 'y', 'z']] = normalized_points
    return df

def find_transformation_matrix(df_source, df_target):
    pca_source = PCA(n_components=3)
    pca_target = PCA(n_components=3)
    pca_source.fit(df_source[['x', 'y', 'z']])
    pca_target.fit(df_target[['x', 'y', 'z']])
    r = pca_target.components_.T @ pca_source.components_
    return r

def merge_point_clouds(output_folder, point_cloud_1, point_cloud_2, color_pc1=None, color_pc2=None):
    print("\033[95mStarting point cloud merge process...\033[0m")
    cc_executable = r"C:\Program Files\CloudCompare\CloudCompare.exe"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "merged_output.ply")

    header1, df1 = read_ply(point_cloud_1)
    header2, df2 = read_ply(point_cloud_2)

    size1 = bounding_box_size(df1)
    size2 = bounding_box_size(df2)

    largest_dimension_1 = size1.max()
    largest_dimension_2 = size2.max()

    if largest_dimension_1 > largest_dimension_2:
        scale_factor = largest_dimension_1 / largest_dimension_2
        df2 = scale_point_cloud(df2, scale_factor)
    else:
        scale_factor = largest_dimension_2 / largest_dimension_1
        df1 = scale_point_cloud(df1, scale_factor)

    df1 = translate_to_origin(df1)
    df2 = translate_to_origin(df2)

    df1 = normalize_point_cloud(df1)
    df2 = normalize_point_cloud(df2)

    r = find_transformation_matrix(df1, df2)
    df2[['x', 'y', 'z']] = df2[['x', 'y', 'z']].dot(r.T)

    if color_pc1:
        color1 = list(map(int, color_pc1.split(',')))
        df1 = color_point_cloud(df1, color1)

    if color_pc2:
        color2 = list(map(int, color_pc2.split(',')))
        df2 = color_point_cloud(df2, color2)

    colored_point_cloud_1 = os.path.join(output_folder, "colored_cloud_1.ply")
    colored_point_cloud_2 = os.path.join(output_folder, "colored_cloud_2.ply")

    write_ply(colored_point_cloud_1, header1, df1)
    write_ply(colored_point_cloud_2, header2, df2)

    align_merge_command = [
        cc_executable,
        "-o", colored_point_cloud_1,
        "-o", colored_point_cloud_2,
        "-ICP",
        "-MIN_ERROR_DIFF", "1e-6",
        "-ITER", "200",
        "-OVERLAP", "50",
        "-RANDOM_SAMPLING_LIMIT", "20000",
        "-MERGE_CLOUDS",
        "-C_EXPORT_FMT", "PLY",
        "-SAVE_CLOUDS", "FILE", output_path
    ]

    try:
        result = subprocess.run(align_merge_command, check=True, capture_output=True, text=True)
        print("Command executed successfully.")
        print("Output:")
        print(result.stdout)
        print(f"Aligned and merged point cloud saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")

    if os.path.exists(output_path):
        print(f"File {output_path} was created successfully.")
    else:
        print(f"File {output_path} was not created.")

def open_project_report():
    overleaf_url = "https://www.overleaf.com/project/your_project_id"  # Replace with your Overleaf project URL
    webbrowser.open(overleaf_url)
    console.print("Opening project report in Overleaf...", style="bold green")

def open_research_survey():
    survey_url = "https://www.surveymonkey.com/r/your_survey_id"  # Replace with your survey URL
    webbrowser.open(survey_url)
    console.print("Opening research survey...", style="bold green")

def open_point_cloud_in_cloudcompare(file_path):
    cc_executable = r"C:\Program Files\CloudCompare\CloudCompare.exe"
    try:
        subprocess.run([cc_executable, file_path], check=True)
        console.print(f"Opened {file_path} in CloudCompare successfully.", style="bold green")
    except subprocess.CalledProcessError as e:
        console.print(f"An error occurred while opening {file_path} in CloudCompare: {e}", style="bold red")
    except FileNotFoundError:
        console.print("CloudCompare executable not found. Please check the path to CloudCompare.", style="bold red")


def main():
    while True:
        display_main_menu()
        choice = input().strip()
        if choice == '1':
            open_code()
        elif choice == '2':
            open_sliders()
        elif choice == '3':
            while True:
                display_functionality_menu()
                func_choice = input().strip()
                if func_choice == '1':
                    video_path = input("Path to the input video file: ")
                    output_folder = input("Directory to save the extracted frames: ")
                    overlap_threshold = float(input("Overlap threshold for frame selection (default is 0.2): ") or 0.2)
                    select_100_distinct_frames(video_path, output_folder, overlap_threshold)
                elif func_choice == '2':
                    baseDir = input("Base directory for the project: ")
                    binDir = input("Directory where AliceVision binaries are located: ")
                    srcImageDir = input("Directory containing the images: ")
                    numImages = int(input("Number of images to process: "))
                    Run_00_CameraInit(baseDir, binDir, srcImageDir)
                    Run_01_FeatureExtraction(baseDir, binDir, numImages)
                    Run_02_ImageMatching(baseDir, binDir)
                    Run_03_FeatureMatching(baseDir, binDir)
                    Run_04_StructureFromMotion(baseDir, binDir)
                    Run_05_PrepareDenseScene(baseDir, binDir)
                    Run_06_CameraConnection(baseDir, binDir)
                    Run_07_DepthMap(baseDir, binDir, numImages, 3)
                    Run_08_DepthMapFilter(baseDir, binDir)
                    Run_09_Meshing(baseDir, binDir)
                    Run_10_MeshFiltering(baseDir, binDir)
                    Run_11_Texturing(baseDir, binDir)
                elif func_choice == '3':
                    output_path = input("Path to the output file where merged point cloud will be saved: ")
                    point_cloud_1 = input("Path to the first point cloud file: ")
                    point_cloud_2 = input("Path to the second point cloud file: ")
                    color_pc1 = input("Color for the first point cloud in R,G,B format (optional): ")
                    color_pc2 = input("Color for the second point cloud in R,G,B format (optional): ")
                    merge_point_clouds(output_path, point_cloud_1, point_cloud_2, color_pc1, color_pc2)
                elif func_choice == '4':
                    file_path = input("Path to the point cloud file to open in CloudCompare: ")
                    open_point_cloud_in_cloudcompare(file_path)
                elif func_choice == '5':
                    break
                elif func_choice == '6':
                    return
                else:
                    console.print("Invalid choice, please try again.", style="bold red")
        elif choice == '4':
            open_project_report()
        elif choice == '5':
            open_research_survey()
        else:
            console.print("Invalid choice, please try again.", style="bold red")

if __name__ == "__main__":
    main()