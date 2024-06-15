import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
from tqdm import tqdm
import pandas as pd
import subprocess
from sklearn.decomposition import PCA


# Functions from video2image script
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


def select_100_distinct_frames(video_path, output_folder, overlap_threshold=0.2, height_threshold=50):
    print("Starting video to images process...")
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
    selected_indices = [frame_indices[i] for i in selected_indices]

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


# Functions from run_images2model script
def SilentMkdir(theDir):
    try:
        os.mkdir(theDir)
    except:
        pass
    return 0


def Run_00_CameraInit(baseDir, binDir, srcImageDir):
    print("\033[95mRunning CameraInit...\033[0m")
    SilentMkdir(baseDir + "/00_CameraInit")
    binName = binDir + "\\aliceVision_cameraInit.exe"
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
    binName = binDir + "\\aliceVision_featureExtraction.exe"
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
    binName = binDir + "\\aliceVision_imageMatching.exe"
    cmdLine = binName
    cmdLine = cmdLine + " --minNbImages 200 --tree "" --maxDescriptors 500 --verboseLevel info --weights "" --nbMatches 50"
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
    binName = binDir + "\\aliceVision_featureMatching.exe"
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
    binName = binDir + "\\aliceVision_incrementalSfm.exe"
    cmdLine = binName
    cmdLine = cmdLine + " --minAngleForLandmark 2.0 --minNumberOfObservationsForTriangulation 2 --maxAngleInitialPair 40.0 --maxNumberOfMatches 0 --localizerEstimator acransac --describerTypes sift --lockScenePreviouslyReconstructed False --localBAGraphDistance 1"
    cmdLine = cmdLine + " --initialPairA "" --initialPairB "" --interFileExtension .ply --useLocalBA True"
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
    binName = binDir + "\\aliceVision_prepareDenseScene.exe"
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
    binName = binDir + "\\aliceVision_cameraConnection.exe"
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
    binName = binDir + "\\aliceVision_depthMapEstimation.exe"
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
    binName = binDir + "\\aliceVision_depthMapFiltering.exe"
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
    binName = binDir + "\\aliceVision_meshing.exe"
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
    binName = binDir + "\\aliceVision_meshFiltering.exe"
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
    binName = binDir + "\\aliceVision_texturing.exe"
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


# Functions from run_merge_clouds script
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


def merge_point_clouds(output_path, point_cloud_1, point_cloud_2, color_pc1=None, color_pc2=None):
    print("\033[95mStarting point cloud merge process...\033[0m")
    cc_executable = r"C:\Program Files\CloudCompare\CloudCompare.exe"
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

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

    colored_point_cloud_1 = os.path.join(output_dir, "colored_cloud_1.ply")
    colored_point_cloud_2 = os.path.join(output_dir, "colored_cloud_2.ply")

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


# Main script that prompts user for inputs and calls the appropriate functions
def main():
    # Video to images for first video
    print("\033[95mPlease provide the following inputs for the first video to images process:\033[0m")
    video_path_1 = input("Path to the first input video file: ")
    output_folder_1 = input("Directory to save the extracted frames for the first video: ")
    overlap_threshold_1 = float(input("Overlap threshold for frame selection (default is 0.2): ") or 0.2)
    height_threshold_1 = float(input("Height threshold to filter frames (default is 50): ") or 50)
    select_100_distinct_frames(video_path_1, output_folder_1, overlap_threshold_1, height_threshold_1)

    # Video to images for second video
    print("\033[95mPlease provide the following inputs for the second video to images process:\033[0m")
    video_path_2 = input("Path to the second input video file: ")
    output_folder_2 = input("Directory to save the extracted frames for the second video: ")
    overlap_threshold_2 = float(input("Overlap threshold for frame selection (default is 0.2): ") or 0.2)
    height_threshold_2 = float(input("Height threshold to filter frames (default is 50): ") or 50)
    select_100_distinct_frames(video_path_2, output_folder_2, overlap_threshold_2, height_threshold_2)

    # Images to model for first set of images
    print("\033[95mPlease provide the following inputs for the first images to model process:\033[0m")
    baseDir_1 = input("Base directory for the first project: ")
    binDir = input("Directory where AliceVision binaries are located: ")
    srcImageDir_1 = output_folder_1
    numImages_1 = int(input("Number of images to process for the first set: "))
    runStep = input("Run step (e.g., runall, run00, run01, ...): ")

    if runStep == "runall":
        Run_00_CameraInit(baseDir_1, binDir, srcImageDir_1)
        Run_01_FeatureExtraction(baseDir_1, binDir, numImages_1)
        Run_02_ImageMatching(baseDir_1, binDir)
        Run_03_FeatureMatching(baseDir_1, binDir)
        Run_04_StructureFromMotion(baseDir_1, binDir)
        Run_05_PrepareDenseScene(baseDir_1, binDir)
        Run_06_CameraConnection(baseDir_1, binDir)
        Run_07_DepthMap(baseDir_1, binDir, numImages_1, 3)
        Run_08_DepthMapFilter(baseDir_1, binDir)
        Run_09_Meshing(baseDir_1, binDir)
        Run_10_MeshFiltering(baseDir_1, binDir)
        Run_11_Texturing(baseDir_1, binDir)
    elif runStep == "run00":
        Run_00_CameraInit(baseDir_1, binDir, srcImageDir_1)
    elif runStep == "run01":
        Run_01_FeatureExtraction(baseDir_1, binDir, numImages_1)
    elif runStep == "run02":
        Run_02_ImageMatching(baseDir_1, binDir)
    elif runStep == "run03":
        Run_03_FeatureMatching(baseDir_1, binDir)
    elif runStep == "run04":
        Run_04_StructureFromMotion(baseDir_1, binDir)
    elif runStep == "run05":
        Run_05_PrepareDenseScene(baseDir_1, binDir)
    elif runStep == "run06":
        Run_06_CameraConnection(baseDir_1, binDir)
    elif runStep == "run07":
        Run_07_DepthMap(baseDir_1, binDir, numImages_1, 3)
    elif runStep == "run08":
        Run_08_DepthMapFilter(baseDir_1, binDir)
    elif runStep == "run09":
        Run_09_Meshing(baseDir_1, binDir)
    elif runStep == "run10":
        Run_10_MeshFiltering(baseDir_1, binDir)
    elif runStep == "run11":
        Run_11_Texturing(baseDir_1, binDir)
    else:
        print("Invalid step provided.")

    # Images to model for second set of images
    print("\033[95mPlease provide the following inputs for the second images to model process:\033[0m")
    baseDir_2 = input("Base directory for the second project: ")
    srcImageDir_2 = output_folder_2
    numImages_2 = int(input("Number of images to process for the second set: "))

    if runStep == "runall":
        Run_00_CameraInit(baseDir_2, binDir, srcImageDir_2)
        Run_01_FeatureExtraction(baseDir_2, binDir, numImages_2)
        Run_02_ImageMatching(baseDir_2, binDir)
        Run_03_FeatureMatching(baseDir_2, binDir)
        Run_04_StructureFromMotion(baseDir_2, binDir)
        Run_05_PrepareDenseScene(baseDir_2, binDir)
        Run_06_CameraConnection(baseDir_2, binDir)
        Run_07_DepthMap(baseDir_2, binDir, numImages_2, 3)
        Run_08_DepthMapFilter(baseDir_2, binDir)
        Run_09_Meshing(baseDir_2, binDir)
        Run_10_MeshFiltering(baseDir_2, binDir)
        Run_11_Texturing(baseDir_2, binDir)
    elif runStep == "run00":
        Run_00_CameraInit(baseDir_2, binDir, srcImageDir_2)
    elif runStep == "run01":
        Run_01_FeatureExtraction(baseDir_2, binDir, numImages_2)
    elif runStep == "run02":
        Run_02_ImageMatching(baseDir_2, binDir)
    elif runStep == "run03":
        Run_03_FeatureMatching(baseDir_2, binDir)
    elif runStep == "run04":
        Run_04_StructureFromMotion(baseDir_2, binDir)
    elif runStep == "run05":
        Run_05_PrepareDenseScene(baseDir_2, binDir)
    elif runStep == "run06":
        Run_06_CameraConnection(baseDir_2, binDir)
    elif runStep == "run07":
        Run_07_DepthMap(baseDir_2, binDir, numImages_2, 3)
    elif runStep == "run08":
        Run_08_DepthMapFilter(baseDir_2, binDir)
    elif runStep == "run09":
        Run_09_Meshing(baseDir_2, binDir)
    elif runStep == "run10":
        Run_10_MeshFiltering(baseDir_2, binDir)
    elif runStep == "run11":
        Run_11_Texturing(baseDir_2, binDir)
    else:
        print("Invalid step provided.")

    # Merging point clouds from both models
    print("\033[95mPlease provide the following inputs for the point cloud merge process:\033[0m")
    output_path = input("Path to the output file where merged point cloud will be saved: ")
    point_cloud_1 = input("Path to the first point cloud file: ")
    point_cloud_2 = input("Path to the second point cloud file: ")
    color_pc1 = input("Color for the first point cloud in R,G,B format (optional): ")
    color_pc2 = input("Color for the second point cloud in R,G,B format (optional): ")

    print("Running point cloud merge process...")
    merge_point_clouds(output_path, point_cloud_1, point_cloud_2, color_pc1, color_pc2)

    print("\033[95mAll processes completed successfully.\033[0m")


if __name__ == "__main__":
    main()
