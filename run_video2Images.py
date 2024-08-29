import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from multiprocessing import Pool, cpu_count



def select_100_distinct_frames(video_path, output_folder, overlap_threshold=0.2):
    def extract_frames(video_path, output_folder, threshold, interval):

        """

        Extract frames from a video and save them as images if similarity is not greater than threshold.



        Args:

            video_path (str): Path to the input video file.

            output_folder (str): Path to the output folder where images will be saved.

            threshold (float): Similarity threshold between frames. Frames with similarity that not greater than this value will be kept.

            interval (int): Interval between frame extraction.

        """

        # Open the video file

        video_capture = cv2.VideoCapture(video_path)

        count = 0

        # Create output folder if it doesn't exist

        import os

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        saved_frames = []

        frame_count = 0
        while True:

            # Read a frame from the video

            success, frame = video_capture.read()

            # Break if no frame is retrieved

            if not success:
                break

            if frame_count == 0:
                saved_frames.append(frame.copy())

                frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")

                cv2.imwrite(frame_filename, frame)

                frame_count += 1

            should_save_frame = True

            if count % interval == 0:

                # Compare current frame with saved frames

                for saved_frame in saved_frames:
                    similarity = compare_frames(saved_frame, frame)
                    if similarity >= threshold:
                        should_save_frame = False

                        break

            # Save frame to collection if necessary

            if should_save_frame and count % interval == 0:

                saved_frames.append(frame.copy())
                frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_filename, frame)
                if success:
                    print(f"Frame {frame_count} saved successfully as {frame_filename}")
                else:
                    print(f"Failed to save frame {frame_count} as {frame_filename}")

                frame_count += 1

            count += 1

        # Release the video capture object

        video_capture.release()

        print("the all frame is :", count / interval)


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video at equal time intervals.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("output_folder", type=str, help="Directory to save the extracted frames.")

    args = parser.parse_args()
    select_100_distinct_frames(args.video_path, args.output_folder)

if __name__ == "__main__":
    main()

    print("Done.")
