import os
import glob
import yaml
import cv2
from PIL import Image
from dlss_enhancer import enhance_images

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def process_images(image_paths, output_dir):
    """Enhances a batch of images and saves them to output_dir."""
    if not image_paths:
        return
    
    print(f"Processing {len(image_paths)} images...")
    results = enhance_images(image_paths)
    
    for src_path, enhanced_img in zip(image_paths, results):
        filename = os.path.basename(src_path)
        name, _ = os.path.splitext(filename)
        out_path = os.path.join(output_dir, f"{name}_enhanced.png")
        enhanced_img.save(out_path)
        print(f"Saved: {out_path}")

def process_video(video_path, output_dir):
    """Extracts frames from a video, enhances them, and reconstructs the video."""
    filename = os.path.basename(video_path)
    name, ext = os.path.splitext(filename)
    out_path = os.path.join(output_dir, f"{name}_enhanced{ext}")
    
    print(f"Processing video: {filename}...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    frame_batch = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR (OpenCV) to RGB (PIL Image)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        frame_batch.append(pil_img)
        
        # Process in batches of 16 frames to prevent memory overflow
        if len(frame_batch) >= 16:
            enhanced_frames = enhance_images(frame_batch)
            for eff in enhanced_frames:
                # Convert back to OpenCV format (RGB -> BGR)
                bgr_frame = cv2.cvtColor(np.array(eff), cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)
            frame_batch.clear()

    # Process remaining frames
    if frame_batch:
        enhanced_frames = enhance_images(frame_batch)
        for eff in enhanced_frames:
            bgr_frame = cv2.cvtColor(np.array(eff), cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)

    cap.release()
    out.release()
    print(f"Saved enhanced video: {out_path}")

def main():
    config = load_config("config.yaml")
    input_dir = config.get("input")
    output_dir = config.get("output")
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_files = glob.glob(os.path.join(input_dir, "*"))
    image_files = []
    video_files = []

    for f in all_files:
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            image_files.append(f)
        elif ext in VIDEO_EXTENSIONS:
            video_files.append(f)

    # Enhance Images
    if image_files:
        process_images(image_files, output_dir)
        
    # Enhance Videos
    for vid in video_files:
        process_video(vid, output_dir)

if __name__ == "__main__":
    import numpy as np
    main()