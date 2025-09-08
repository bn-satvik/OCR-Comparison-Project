import os
from paddleocr import PaddleOCR
from PIL import Image, ImageOps

# Create OCR object
ocr = PaddleOCR(lang='en', use_textline_orientation=True)

# Paths
input_path = "assets/image.png"  # Change to your image file
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

def preprocess_image(image_path):
    img = Image.open(image_path)
    # Upscale 2x
    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    # Convert to grayscale
    img = ImageOps.grayscale(img)
    # Binarize (threshold)
    img = img.point(lambda x: 0 if x < 180 else 255, '1')
    processed_path = os.path.join(output_dir, "processed.png")
    img.save(processed_path)
    return processed_path

def run_ocr_on_image(image_path):
    results = ocr.predict(image_path)
    text_lines = []
    for line in results[0]:
        if isinstance(line[1], tuple):
            text, conf =line[1]
            text_lines.append(f"{text} ({conf:.2f})")
        else:
            text_lines.append(str(line[1]))
    return text_lines

print(f"Processing image: {input_path}")
processed_image = preprocess_image(input_path)
text = run_ocr_on_image(processed_image)

# Save results
output_file = os.path.join(output_dir, "result.txt")
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(text))

print(f"OCR complete. Results saved in {output_file}")