import os
import json
import fitz  # PyMuPDF
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Config: 1 = image, 0 = PDF
env_pdf_or_image = 0

# Paths
input_path = "assets/OCRtestCase1.pdf"  # Change to your file
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Load DocTR OCR model
model = ocr_predictor(pretrained=True)

def extract_text_with_boxes_doctr(image_path):
    doc = DocumentFile.from_images(image_path)
    result = model(doc)
    export = result.export()

    output = []
    for page in export['pages']:
        for block in page['blocks']:
            for line in block['lines']:
                for word in line['words']:
                    text = word['value']
                    # bbox is relative [x_min, y_min, x_max, y_max] in 0-1 range
                    rel_bbox = word['geometry']
                    output.append({
                        "text": text,
                        "bounding_box": rel_bbox  # normalized coordinates
                    })
    return output

if env_pdf_or_image == 1:
    # Process image
    print(f"Processing image: {input_path}")
    results = extract_text_with_boxes_doctr(input_path)
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

        page_results = extract_text_with_boxes_doctr(img_path)
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