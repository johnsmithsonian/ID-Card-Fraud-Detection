import os
import cv2

def save_collage_image(collage, filename):
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    collage_path = os.path.join(results_dir, filename)
    cv2.imwrite(collage_path, collage)
    return collage_path

def load_results_images():
    results_dir = 'results'
    images = []
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            if filename.endswith('.jpg'):
                img_path = os.path.join(results_dir, filename)
                images.append(img_path)
    return images
