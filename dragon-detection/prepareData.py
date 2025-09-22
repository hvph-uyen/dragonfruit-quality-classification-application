import os
import shutil
import random
from PIL import Image
import cv2
import numpy as np
from rembg import remove

def prepare_detector_dataset(classifier_dir="dataset", detector_dir="dataset2"):
    """
    Merge 'good', 'immature', 'reject' into one 'dragonfruit' folder for detector dataset.
    Keeps not_dragonfruit untouched.
    """
    for split in ["train", "val"]:
        # Source dirs (classifier dataset)
        classifier_split_dir = os.path.join(classifier_dir, split)
        dragonfruit_out_dir = os.path.join(detector_dir, split, "dragonfruit")
        os.makedirs(dragonfruit_out_dir, exist_ok=True)

        # Merge good, immature, reject
        for cls in ["good", "immature", "reject"]:
            src_dir = os.path.join(classifier_split_dir, cls)
            if not os.path.exists(src_dir):
                print(f"Skipping missing {src_dir}")
                continue
            for fname in os.listdir(src_dir):
                src_path = os.path.join(src_dir, fname)
                dst_path = os.path.join(dragonfruit_out_dir, f"{cls}_{fname}")
                shutil.copy2(src_path, dst_path)

        print(f"Created {dragonfruit_out_dir} with merged images.")

    print("Detector dataset preparation complete!")

def split_dataset(
    source_dir, 
    output_dir='split_good', 
    train_ratio=0.8, 
    seed=42
):
    # Ensure reproducibility
    random.seed(seed)

    # Get all image files
    all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    random.shuffle(all_files)

    # Calculate split index
    split_idx = int(len(all_files) * train_ratio)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    # Define output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Move files
    for filename in train_files:
        shutil.copy(os.path.join(source_dir, filename), os.path.join(train_dir, filename))
    for filename in val_files:
        shutil.copy(os.path.join(source_dir, filename), os.path.join(val_dir, filename))

    print(f"Total images: {len(all_files)}")
    print(f"Training: {len(train_files)}")
    print(f"Validation: {len(val_files)}")
    print(f"Dataset split completed. Output directory: '{output_dir}'")

def resize_images(
    source_dir, 
    output_dir='resized', 
    size=(224, 224), 
    keep_aspect_ratio=False
):
    os.makedirs(output_dir, exist_ok=True)

    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    for filename in os.listdir(source_dir):
        if not filename.lower().endswith(supported_formats):
            continue  # skip non-image files

        input_path = os.path.join(source_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            with Image.open(input_path) as img:
                if keep_aspect_ratio:
                    img.thumbnail(size, Image.Resampling.LANCZOS)
                else:
                    img = img.resize(size, Image.Resampling.LANCZOS)

                img.save(output_path)
                print(f"Resized and saved: {output_path}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    print(f"All images resized and saved to: '{output_dir}'")

def crop_with_grabcut(img_path, out_path):
    img = cv2.imread(img_path)
    mask = np.zeros(img.shape[:2], np.uint8)

    # Model arrays for GrabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Rectangle around center (adjust if fruit is small/large)
    h, w = img.shape[:2]
    rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))

    # Apply GrabCut
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    img_cropped = img * mask2[:, :, np.newaxis]

    # Bounding box around foreground
    ys, xs = np.where(mask2 == 1)
    if len(xs) > 0 and len(ys) > 0:
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        img_cropped = img_cropped[y1:y2, x1:x2]

    cv2.imwrite(out_path, img_cropped)

if __name__ == "__main__":
    #prepare_detector_dataset()
    #split_dataset(source_dir='good', train_ratio=0.8)
    #resize_images(source_dir='reject', output_dir='resized_reject', size=(416, 416))

    # crop image background
    INPUT_DIR = "temp"
    OUTPUT_DIR = "temp_cropped"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for fname in os.listdir(INPUT_DIR):
        inp = Image.open(os.path.join(INPUT_DIR, fname))
        out = remove(inp)  # removes background
        fname = os.path.splitext(fname)[0] + ".png"
        out.save(os.path.join(OUTPUT_DIR, fname), "PNG")
