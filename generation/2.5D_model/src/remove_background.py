import os
import cv2
import numpy as np
import tqdm


def process_image(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # Keep original grayscale values
    gray = image

    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"No contours found in: {image_path}")
        return

    mask = np.zeros_like(binary)

    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    alpha = mask
    rgba = cv2.merge([gray, gray, gray, alpha])

    outline = np.zeros_like(gray)
    cv2.drawContours(outline, [largest_contour], -1, 255, thickness=2)
    outline_rgba = cv2.merge([outline, outline, outline, outline])
    rgba[outline_rgba[:, :, 0] == 255] = [255, 255, 255, 255]

    cv2.imwrite(output_path, rgba)


def batch_process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(input_dir) if f.endswith(".png") and "Process" not in f]
    print(f"Processing {len(image_files)} images...")

    for filename in tqdm.tqdm(image_files):
        try:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            process_image(input_path, output_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("Processing complete.")
