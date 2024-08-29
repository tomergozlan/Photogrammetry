import os
import pandas as pd
import subprocess
from sklearn.decomposition import PCA


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
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
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
        "-PLY_EXPORT_FMT", "ASCII",  # Ensure ASCII format
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


def main():
    point_cloud_1 = input("Enter the path for the first point cloud file: ")
    point_cloud_2 = input("Enter the path for the second point cloud file: ")
    output_folder = input("Enter the path to the folder where the output should be saved: ")

    color_pc1 = input("Enter the color for the first point cloud (as R,G,B) or leave blank for default: ")
    color_pc2 = input("Enter the color for the second point cloud (as R,G,B) or leave blank for default: ")

    if not color_pc1:
        color_pc1 = None
    if not color_pc2:
        color_pc2 = None

    merge_point_clouds(output_folder, point_cloud_1, point_cloud_2, color_pc1, color_pc2)


if __name__ == "__main__":
    main()
