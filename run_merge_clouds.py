import argparse
import pandas as pd
import subprocess
import os
from sklearn.decomposition import PCA


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


def main():
    parser = argparse.ArgumentParser(description="CLI tool for combining two PointClouds")
    parser.add_argument("output_path", type=str, help="Path to the output file where merged point cloud will be saved")
    parser.add_argument("point_cloud_1", type=str, help="Path to the first point cloud file")
    parser.add_argument("point_cloud_2", type=str, help="Path to the second point cloud file")
    parser.add_argument("--color_pc1", type=str, help="Color for the first point cloud in R,G,B format "
                                                      "(e.g., 255,0,0 for red)", default=None)
    parser.add_argument("--color_pc2", type=str, help="Color for the second point cloud in R,G,B format "
                                                      "(e.g., 0,255,0 for green)", default=None)
    args = parser.parse_args()

    cc_executable = r"C:\Program Files\CloudCompare\CloudCompare.exe"
    output_path = args.output_path
    point_cloud_1 = args.point_cloud_1
    point_cloud_2 = args.point_cloud_2

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

    # Color the point clouds if specified
    if args.color_pc1:
        color1 = list(map(int, args.color_pc1.split(',')))
        df1 = color_point_cloud(df1, color1)

    if args.color_pc2:
        color2 = list(map(int, args.color_pc2.split(',')))
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


if __name__ == "__main__":
    main()
