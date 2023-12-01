import os
import cv2
import numpy as np

def salt_noise(image, salt_prob):
    noisy_image = np.copy(image)
    total_pixels = image.size // 3  # Divide by 3 for RGB images

    num_salt = np.ceil(salt_prob * total_pixels).astype(int)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 255

    return noisy_image

def get_images(path):
    image_extensions = [".jpg", ".jpeg", ".png"]
    return [file for file in os.listdir(path) if os.path.splitext(file)[1].lower() in image_extensions]


def salt_levels_generation():
    levels = {'original_image.jpg': 0, 'noisy_image_level1.jpg': 0.05, 'noisy_image_level2.jpg': 0.07, 'noisy_image_level3.jpg': 0.1}

    original_images_folder = "original-images"
    image_paths = get_images(original_images_folder)

    for image_path in image_paths:
        image = cv2.imread(os.path.join(original_images_folder, image_path))
        image_name = image_path.split(".")[0]
        output_folder = f"noisy-images/{image_name}"

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for image_name, level in levels.items():
            noisy_image = salt_noise(image, level)
            cv2.imwrite(os.path.join(output_folder, image_name), noisy_image)

salt_levels_generation()