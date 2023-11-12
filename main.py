import cv2
import numpy as np
import pytesseract

def crop_license_plate(gray_plate):
    blurred = cv2.GaussianBlur(gray_plate, (5, 5), 0)
    _, binarized = cv2.threshold(cv2.equalizeHist(blurred), 150, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(binarized, 100, 200)
    cv2.imshow("binarized", binarized)
    cv2.imshow("edges", edges)

    cv2.imwrite("generated/binarized.jpg", binarized)
    cv2.imwrite("generated/edges.jpg", edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    cropped_images = []

    for countor in contours:
        x, y, w, h = cv2.boundingRect(countor)
        cropped = gray_plate[y:y+h, x:x+w]
        cropped_images.append(cropped)
        cv2.imwrite("generated/crops-" + str(x) + ".jpg", cropped)


    return cropped_images

def segment_into_characters(gray_plate):
    all_segments = []

    cropped_images = crop_license_plate(gray_plate)

    for i, cropped in enumerate(cropped_images):
        blurred = cv2.bilateralFilter(cropped, 5, 17, 17)
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow(str(i), thresh)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
            cv2.imwrite("generated/raw-" + str(x) + ".jpg", segment)
            _, binarized = cv2.threshold(segment, 140, 255, cv2.THRESH_BINARY)
            padded_image = cv2.copyMakeBorder(binarized, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255])
            segments.append(padded_image)
            cv2.imwrite("generated/edited-" + str(x) + ".jpg", padded_image)

        all_segments.append(segments)

    return all_segments


def salt_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    total_pixels = image.size // 3  # Divide by 3 for RGB images

    num_salt = np.ceil(salt_prob * total_pixels).astype(int)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 255

    num_pepper = np.ceil(pepper_prob * total_pixels).astype(int)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 0

    return noisy_image

def median_filter(image, kernel_size):
    filtered_image = cv2.medianBlur(image, kernel_size)
    return filtered_image

def main():
    image = cv2.imread('noisy.jpg')
    #image = salt_pepper_noise(image, 0.05, 0.03)

    #cv2.imwrite("generated/noisy.jpg", image)
    cv2.imshow("noisy", image)
    image = median_filter(image, 3)

    cv2.imshow("median", image)
    cv2.imwrite("generated/median.jpg", image)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
    plates = plate_cascade.detectMultiScale(image, 1.1, 40)

    for (x, y, w, h) in plates:
        cv2.imwrite("detected-license-plate.jpg", image[y:y+h, x:x+w])
        gray_plate = gray_image[y:y+h, x:x+w]
        all_segments = segment_into_characters(gray_plate)
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

        print("Detected license plate:", "".join(longest_chars));

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Detected Plates', image)
    cv2.imwrite("generated/detection.jpg", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main_for_noise():
    image = cv2.imread('car8.jpg')
    noisy_image = salt_pepper_noise(image, 0.05, 0.05)
    cv2.imshow("noisy", noisy_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main_binarizer():
    image = cv2.imread('graph.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
    cv2.imwrite("binarized.jpg", binarized)

main()