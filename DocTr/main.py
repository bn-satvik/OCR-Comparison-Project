
# --- Updated for batch evaluation ---
import os
import json
import time
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image

def cer(s1, s2):
    """Character Error Rate"""
    import difflib
    s1, s2 = s1.replace("\n", " "), s2.replace("\n", " ")
    matcher = difflib.SequenceMatcher(None, s1, s2)
    errors = sum([max(a2 - a1, b2 - b1) for (tag, a1, a2, b1, b2) in matcher.get_opcodes() if tag != 'equal'])
    return errors / max(1, len(s2)) * 100

def wer(s1, s2):
    """Word Error Rate"""
    import difflib
    w1, w2 = s1.split(), s2.split()
    matcher = difflib.SequenceMatcher(None, w1, w2)
    errors = sum([max(a2 - a1, b2 - b1) for (tag, a1, a2, b1, b2) in matcher.get_opcodes() if tag != 'equal'])
    return errors / max(1, len(w2)) * 100

def doctr_to_text(image_path):
    doc = DocumentFile.from_images(image_path)
    result = model(doc)
    export = result.export()
    lines = []
    for page in export['pages']:
        for block in page['blocks']:
            for line in block['lines']:
                line_text = " ".join([w['value'] for w in line['words']])
                lines.append(line_text)
    return "\n".join(lines)

def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size


def process_subfolder(subfolder):
    image_files = [f for f in os.listdir(subfolder) if f.endswith('.png')]
    results = []
    print(f"\nProcessing folder: {subfolder}")
    print(f"Found {len(image_files)} images in {subfolder}")

    total_time = 0.0
    total_cer = 0.0
    total_wer = 0.0
    total_height = 0
    total_width = 0
    count = 0

    for img_file in image_files:
        img_path = os.path.join(subfolder, img_file)
        txt_file = img_file.replace('.png', '.txt')
        txt_path = os.path.join(subfolder, txt_file)
        if not os.path.exists(txt_path):
            print(f"Skipping {img_file}: ground-truth {txt_file} not found.")
            continue

        gt_text = open(txt_path, encoding='utf-8').read().strip()
        width, height = get_image_size(img_path)

        start = time.time()
        ocr_text = doctr_to_text(img_path)
        elapsed = time.time() - start

        cer_val = cer(ocr_text, gt_text)
        wer_val = wer(ocr_text, gt_text)

        results.append({
            'image': img_file,
            'inference_time': elapsed,
            'cer': cer_val,
            'wer': wer_val,
            'height': height,
            'width': width
        })

        total_time += elapsed
        total_cer += cer_val
        total_wer += wer_val
        total_height += height
        total_width += width
        count += 1
        print(f"{img_file}: time={elapsed:.3f}s, CER={cer_val:.2f}%, WER={wer_val:.2f}%")

    if count == 0:
        print("No images processed in folder.")
        return

    avg_time = total_time / count
    avg_cer = total_cer / count
    avg_wer = total_wer / count
    avg_height = total_height / count
    avg_width = total_width / count

    print("\n| model       | average_inference_time | average_cer | average_wer | image_count | average_image_height | average_image_width |")
    print("|:-----------|:----------------------|:------------|:------------|------------:|---------------------:|--------------------:|")
    print(f"| DocTR       | {avg_time:.4f}s               | {avg_cer:.2f}%      | {avg_wer:.2f}%      | {count:11d} | {avg_height:20.0f} | {avg_width:18.0f} |")

def main():
    assets_dir = "assets"
    subfolders = [os.path.join(assets_dir, d) for d in os.listdir(assets_dir) if os.path.isdir(os.path.join(assets_dir, d))]
    if not subfolders:
        print("No subfolders found in assets.")
        return
    for subfolder in subfolders:
        process_subfolder(subfolder)

# Load DocTR OCR model (outside main for speed)
model = ocr_predictor(pretrained=True)

if __name__ == "__main__":
    main()
