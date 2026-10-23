
import cv2
import os

# ==============================
# Configuration
# ==============================

VIDEO_PATH = r"C:\Users\YourName\Videos\your_video.mp4"
OUTPUT_FOLDER = "training_frames"
FRAME_INTERVAL = 60  # Extract one frame every 60 frames


def extract_frames(video_path, output_folder, frame_interval):
    # Check if the video exists
    if not os.path.isfile(video_path):
        print(f"Error: Video not found: {video_path}")
        return

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: Could not open the video.")
        return

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)

    print(f"Video FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Extracting one frame every {frame_interval} frames...")

    frame_number = 0
    saved_count = 0

    while True:
        success, frame = video.read()

        if not success:
            break

        # Save frames at 0, 60, 120, 180, ...
        if frame_number % frame_interval == 0:
            filename = f"frame_{frame_number:06d}.jpg"
            output_path = os.path.join(output_folder, filename)

            cv2.imwrite(output_path, frame)
            saved_count += 1

            print(f"Saved: {output_path}")

        frame_number += 1

    video.release()

    print("\nExtraction completed!")
    print(f"Frames processed: {frame_number}")
    print(f"Images saved: {saved_count}")
    print(f"Output folder: {os.path.abspath(output_folder)}")


if __name__ == "__main__":
    extract_frames(
        VIDEO_PATH,
        OUTPUT_FOLDER,
        FRAME_INTERVAL
    )
