"""Convert annotated MP4 videos to GIFs for README demos.

Usage:
    python scripts/video_to_gif.py
"""

import cv2
from PIL import Image
from pathlib import Path

VIDEOS = {
    "outputs/chest fly machine_1_annotated.mp4": "assets/chest_fly_machine.gif",
    "outputs/leg extension_1_annotated.mp4": "assets/leg_extension.gif",
}

# GIF settings
TARGET_WIDTH = 320
FRAME_SKIP = 4  # take every Nth frame (reduces file size)
MAX_FRAMES = 60  # cap total GIF frames
GIF_FPS = 8


def video_to_gif(video_path: str, gif_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return

    frames = []
    count = 0
    while len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % FRAME_SKIP != 0:
            continue

        # Resize
        h, w = frame.shape[:2]
        scale = TARGET_WIDTH / w
        new_h = int(h * scale)
        frame = cv2.resize(frame, (TARGET_WIDTH, new_h))

        # BGR → RGB for Pillow
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()

    if not frames:
        print(f"  ERROR: No frames extracted from {video_path}")
        return

    # Save GIF
    Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
    duration = int(1000 / GIF_FPS)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
    size_mb = Path(gif_path).stat().st_size / (1024 * 1024)
    print(f"  {gif_path} ({len(frames)} frames, {size_mb:.1f} MB)")


def main():
    print("Converting videos to GIFs...\n")
    for video, gif in VIDEOS.items():
        print(f"  {Path(video).name} → {gif}")
        video_to_gif(video, gif)
    print("\nDone!")


if __name__ == "__main__":
    main()
