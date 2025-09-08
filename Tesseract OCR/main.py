import os
import json
import pytesseract
from PIL import Image
import fitz  # PyMuPDF

# Config: 1 = image, 0 = PDF
env_pdf_or_image = 1  

# Paths
input_path = "assets/image(1).png"  # Change to your file
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

def extract_text_with_boxes(image_path):
    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    results = []
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if text:
            x1, y1 = data['left'][i], data['top'][i]
            x2, y2 = x1 + data['width'][i], y1 + data['height'][i]
            results.append({
                "text": text,
                "bounding_box": [x1, y1, x2, y2]
            })
    return results

if env_pdf_or_image == 1:
    # Process image
    print(f"Processing image: {input_path}")
    results = extract_text_with_boxes(input_path)
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

        page_results = extract_text_with_boxes(img_path)
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