# License plate recognition

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Python**: This project requires Python to be installed. If you don't have Python installed, download and install it from [python.org](https://www.python.org/downloads/).

- **OpenCV (cv2)**: OpenCV is used for image processing. After installing Python, you can install OpenCV by running `pip install opencv-python` in your terminal.

- **NumPy**: NumPy is a library for numerical computations. Install it using `pip install numpy`.

- **Tesseract OCR**: Tesseract is an optical character recognition engine. Install it by following the instructions on the [Tesseract GitHub page](https://github.com/tesseract-ocr/tesseract).

## Installation

To set up the project, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the project directory in your terminal.

## Preparing Images

Before running the main script, ensure you do the following:

1. Place your car images in the `original-images` folder.
2. Run following command to apply salt and pepper noise to images and save noisy images in `noisy-images` folder.

```bash
python generate_noisy_images.py
```

## Usage

To run the main script, use the following command in your terminal:

```bash
python main.py --char-transformation "erode"
```

### Options for `--char-transformation`:

- `dilate`: Apply dilation morphological operation to segmented characters.
- `erode`: Apply erosion morphological operation.
- `none`: No character transformation will be applied.

Adjust the `--char-transformation` argument based on your requirement.
