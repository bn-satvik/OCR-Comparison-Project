import os
import json
import easyocr
from PIL import Image
import fitz  # PyMuPDF

# Config: 1 = image, 0 = PDF
env_pdf_or_image = 0

# Paths
input_path = "assets/OCRtestCase1.pdf"  # Change to your file
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Create EasyOCR reader
reader = easyocr.Reader(['en'])

def extract_text_with_boxes_easyocr(image_path):
    results = reader.readtext(image_path)
    output = []
    for bbox, text, conf in results:
        # bbox is a list of 4 points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)
        output.append({
            "text": text,
            "bounding_box": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": round(conf, 2)
        })
    return output

if env_pdf_or_image == 1:
    # Process image
    print(f"Processing image: {input_path}")
    results = extract_text_with_boxes_easyocr(input_path)
    output_file = os.path.join(output_dir, "result_image.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"OCR complete. Results saved in {output_file}")

elif env_pdf_or_image == 0:
    # Process PDF
    print(f"Processing PDF: {input_path}")
    doc = fitz.open(input_path)
    all_results = []

    for page_num in range(len(doc)):
        pix = doc[page_num].get_pixmap()
        img_path = os.path.join(output_dir, f"page_{page_num+1}.png")
        pix.save(img_path)

        page_results = extract_text_with_boxes_easyocr(img_path)
        all_results.append({
            "page": page_num + 1,
            "results": page_results
        })

    output_file = os.path.join(output_dir, "result_pdf.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"OCR complete. Results saved in {output_file}")

else:
    print("Invalid value for env_pdf_or_image. Use 1 for image, 0 for PDF.")