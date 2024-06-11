# Import required packages
import easyocr
import cv2
import argparse
import difflib

ap = argparse.ArgumentParser()
ap.add_argument("--i", required=True, help="name of the image to be OCR'd")

args = vars(ap.parse_args())

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Resize the image to increase resolution
    scale_percent = 200  # percent of original size
    width = int(blurred_image.shape[1] * scale_percent / 100)
    height = int(blurred_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(blurred_image, dim, interpolation=cv2.INTER_CUBIC)

    # Apply adaptive thresholding
    processed_image = cv2.adaptiveThreshold(
        resized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    return processed_image

def preform_ocr(img):
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(img)
    return results

img = preprocess_image("./samples/" + args["i"])

words = []

# Print results
for (bbox, text, prob) in preform_ocr(img):
    if prob > 0.5:
        print(f"EASYOCR Detected text: {text} (Confidence: {prob})")
        words.append(text)

# Check Manufacturers
manufacturers = ["agl", "clash", "discmania", 
                "dynamic", "finish line", 
                 "innova", "legacy", "millennium", 
                 "prodigy", "thought space", "yikun", 
                 "axiom", "dga", "discraft", 
                 "elevation", "gateway", "kastaplast", 
                 "loft", "mint", "rpm", "trash panda", 
                 "birdie", "hooligan", "divergent", "ev-7", 
                 "infinite", "latitude 64", "lone star", "mvp", 
                 "streamline", "westside"]

def manufacturers_similarities(word):
    for man in manufacturers:
        if difflib.SequenceMatcher(a=word.lower(), b=man.lower()).ratio() > 0.9:
            return man
    return ""

for word in words:
    potential = manufacturers_similarities(word)
    if not potential == "":
        print(f"{potential}")

print(f"{words}")