import cv2
import sys
import os
import numpy as np
import pytesseract

def crop_license_plate(gray_plate):
    blurred = cv2.GaussianBlur(gray_plate, (5, 5), 0)
    _, binarized = cv2.threshold(cv2.equalizeHist(blurred), 150, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    cropped_images = []

    for countor in contours:
        x, y, w, h = cv2.boundingRect(countor)
        cropped = gray_plate[y:y+h, x:x+w]
        cropped_images.append(cropped)


    return cropped_images

def segment_into_characters(gray_plate, char_transformation):
    all_segments = []

    cropped_images = crop_license_plate(gray_plate)

    for i, cropped in enumerate(cropped_images):
        blurred = cv2.bilateralFilter(cropped, 5, 17, 17)
        _, binarized = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        char_candidates = []

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            aspect_ratio = w / float(h)
            solidity = cv2.contourArea(contour) / float(w * h)

            if aspect_ratio < 1 and solidity > 0.1:
                char_candidates.append((x, y, w, h))

        char_candidates = sorted(char_candidates, key=lambda x: x[0])

        segments = []
        for (x, y, w, h) in char_candidates:
            segment = cropped[y:y+h, x:x+w]

            _, binarized = cv2.threshold(segment, 140, 255, cv2.THRESH_BINARY)
            enhanced_segment = cv2.copyMakeBorder(binarized, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255])
        
            enhanced_segment = cv2.bitwise_not(enhanced_segment)

            kernel_size = 3
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            if char_transformation == "erode":
                enhanced_segment = cv2.erode(enhanced_segment, kernel, iterations=1)
            elif char_transformation == "dilate":
                enhanced_segment = cv2.dilate(enhanced_segment, kernel, iterations=1)

            enhanced_segment = cv2.bitwise_not(enhanced_segment)
            segments.append(enhanced_segment)

        all_segments.append(segments)

    return all_segments


def median_filter(image, kernel_size):
    filtered_image = cv2.medianBlur(image, kernel_size)
    return filtered_image

def recognize_license_plate(path, char_transformation):
    image = cv2.imread(path)
    image = median_filter(image, 3)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
    plates = plate_cascade.detectMultiScale(image, 1.1, 40)

    path_title = f"recognized license plate of {path}: "

    if len(plates) == 0:
        print(path_title, "Not found")

    for (x, y, w, h) in plates:
        gray_plate = gray_image[y:y+h, x:x+w]

        all_segments = segment_into_characters(gray_plate, char_transformation)
        longest_chars = []

        for segments in all_segments:
            chars = []

            for segment in segments:
                config = '-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 10'
                char = pytesseract.image_to_string(segment, config=config)
                if len(char) > 0:
                    chars.append(char[0])

            if len(chars) > len(longest_chars):
                longest_chars = chars

        print(path_title, "".join(longest_chars))

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 6)

        recognized_license_plate = "".join(longest_chars)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_color = (0, 255, 0)
        line_thickness = 6
        cv2.putText(image, recognized_license_plate, (x, y + h + 60), font, font_scale, font_color, line_thickness)

    cv2.imshow(path, image)

def get_directories(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def get_images(path):
    image_extensions = [".jpg", ".jpeg", ".png"]
    return [file for file in os.listdir(path) if os.path.splitext(file)[1].lower() in image_extensions]


def main():
    # variants: "erode", "dilate" or "none"
    char_transformation = "none"

    if "--char-transformation" in sys.argv:
        index = sys.argv.index("--char-transformation") + 1
        char_transformation = sys.argv[index] if index < len(sys.argv) else "none"

    noisy_images_folder = "noisy-images"
    directories = get_directories(noisy_images_folder)

    for dir in directories:
      print("-------------------------------------------------")

      image_paths = get_images(os.path.join(noisy_images_folder, dir))
      image_paths.sort()

      for image_path in image_paths:
        recognize_license_plate(os.path.join(noisy_images_folder, dir, image_path), char_transformation)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()