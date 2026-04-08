from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


BORDER_SIZE = 50
CROP_LENGTH = 200
OUTPUT_SIZE = (300, 300)
QR_RATIO = 0.12


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """支援 Windows 中文路徑讀圖"""
    path = str(path)
    if not Path(path).exists():
        return None
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path, image):
    """支援 Windows 中文路徑寫圖"""
    path = str(path)
    ext = Path(path).suffix
    if ext == "":
        ext = ".png"

    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False

    encoded.tofile(path)
    return True


def read_unicode_list(json_path, unicode_num):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cp950 = data["CP950"][:unicode_num]
    return [f"U+{item['UNICODE'][2:6]}" for item in cp950]


def ensure_clean_dir(directory):
    directory = Path(directory)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def make_unique_filename(base_name, used_names):
    used_names[base_name] += 1
    count = used_names[base_name]
    if count == 1:
        return f"{base_name}.png"
    return f"{base_name}_{count}.png"


def remove_small_components(image, min_area_threshold):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        image, connectivity=8
    )
    cleaned = image.copy()
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area < min_area_threshold:
            cleaned[labels == label_idx] = 255
    return cleaned


def preprocess_cropped_image(cropped_gray, min_area_threshold):
    median_filtered = cv2.medianBlur(cropped_gray, 3)
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(median_filtered, cv2.MORPH_OPEN, kernel)
    cleaned = remove_small_components(opened, min_area_threshold)
    return cleaned


def scale_adjustment(word_img, img_name, save_annotated=True):
    word_img_copy = cv2.copyMakeBorder(
        word_img,
        BORDER_SIZE,
        BORDER_SIZE,
        BORDER_SIZE,
        BORDER_SIZE,
        cv2.BORDER_CONSTANT,
        value=255,
    )

    if len(word_img_copy.shape) == 3:
        gray = cv2.cvtColor(word_img_copy, cv2.COLOR_BGR2GRAY)
    else:
        gray = word_img_copy

    binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]

    if cv2.countNonZero(binary) == 0:
        return cv2.resize(word_img_copy, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)

    x, y, w, h = cv2.boundingRect(binary)
    c_x, c_y = x + w // 2, y + h // 2

    if save_annotated:
        annotated_dir = Path("annotated_images")
        annotated_dir.mkdir(parents=True, exist_ok=True)

        annotated_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (255, 168, 0), 2)
        cv2.circle(annotated_img, (c_x, c_y), 6, (0, 0, 255), -1)
        imwrite_unicode(annotated_dir / f"{img_name}_annotated.png", annotated_img)

    half = CROP_LENGTH // 2
    img_h, img_w = word_img_copy.shape[:2]

    left = max(0, c_x - half)
    right = min(img_w, c_x + half)
    top = max(0, c_y - half)
    bottom = min(img_h, c_y + half)

    final_word_img = word_img_copy[top:bottom, left:right]
    return cv2.resize(final_word_img, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)


def extract_page_number(path):
    name = path.name
    patterns = [
        r"page-(\d+)_qr-(\d+)\.png$",
        r"page-(\d+)\.png$",
    ]

    for pattern in patterns:
        match = re.match(pattern, name, re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None


def collect_page_images(input_folder, start_page, end_page):
    input_folder = Path(input_folder)
    candidates = []

    for path in input_folder.glob("*.png"):
        page_num = extract_page_number(path)
        if page_num is None:
            continue
        if start_page <= page_num <= end_page:
            candidates.append((page_num, path))

    candidates.sort(key=lambda x: x[0])
    return candidates


def find_boxes_from_page(img_color, debug_name="debug"):
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)[1]

    img_h, img_w = binary.shape
    qr_size = int(min(img_h, img_w) * QR_RATIO)
    binary[-qr_size:, -qr_size:] = 0

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    debug_dir = Path("debug_binary")
    debug_dir.mkdir(parents=True, exist_ok=True)
    imwrite_unicode(debug_dir / f"{debug_name}_binary.png", binary)

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w < 120 or h < 120:
            continue

        ratio = w / float(h)
        if 0.7 <= ratio <= 1.3:
            boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: (b[1] // 120, b[0]))
    return boxes


def crop_boxes(
    input_folder,
    output_folder,
    start_page,
    end_page,
    min_box_size,
    padding,
    json_path,
    unicode_num,
    min_area_threshold,
):
    input_folder = Path(input_folder)
    output_folder = ensure_clean_dir(output_folder)

    bound_output_directory = Path("rec_bound")
    bound_output_directory.mkdir(parents=True, exist_ok=True)

    unicode_list = read_unicode_list(json_path, unicode_num)
    used_names = defaultdict(int)

    page_images = collect_page_images(input_folder, start_page, end_page)

    if not page_images:
        print("找不到符合頁數範圍的圖片。")
        print(f"input_folder = {input_folder}")
        return

    print(f"共找到 {len(page_images)} 張頁面圖片")

    k = (start_page - 1) * 100
    total_saved = 0

    for page_num, image_path in page_images:
        print(f"\nProcessing page {page_num}: {image_path}")

        img_color = imread_unicode(image_path, cv2.IMREAD_COLOR)
        if img_color is None:
            print(f"Warning: cannot read image: {image_path}")
            continue

        boxes = find_boxes_from_page(img_color, debug_name=f"page_{page_num}")
        print(f"  detected boxes: {len(boxes)}")

        if not boxes:
            print("  沒有偵測到可用方框")
            continue

        page_saved = 0

        for (x, y, w, h) in boxes:
            if k >= unicode_num:
                break

            x2 = x + padding
            y2 = y + padding
            w2 = w - 2 * padding
            h2 = h - 2 * padding

            if w2 <= 0 or h2 <= 0:
                continue
            if w2 < min_box_size or h2 < min_box_size:
                continue

            cropped = img_color[y2:y2 + h2, x2:x2 + w2]
            if cropped.size == 0:
                continue

            cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            processed = preprocess_cropped_image(cropped_gray, min_area_threshold)
            final_img = scale_adjustment(processed, unicode_list[k])

            filename = make_unique_filename(unicode_list[k], used_names)
            out_path = output_folder / filename
            ok = imwrite_unicode(out_path, final_img)

            if ok:
                total_saved += 1
                page_saved += 1
                cv2.rectangle(img_color, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)
                k += 1
            else:
                print(f"  write failed: {out_path}")

        imwrite_unicode(bound_output_directory / f"page-{page_num}.png", img_color)
        print(f"  saved on this page: {page_saved}")

    print(f"\n完成裁切，總共輸出 {total_saved} 張字圖")
    print(f"輸出資料夾：{output_folder.resolve()}")


if __name__ == "__main__":
    input_folder = r"c:\Users\ChengYu\Desktop\ai2026s\ai2026s\hw02\02-1_crop_paper\rotated_113590051_楊承諭_千字文"
    output_folder = r"crop\crop_千字文"

    start_page = int(input("Enter start page: "))
    end_page = int(input("Enter end page: "))

    min_box_size = 120
    min_area_threshold = 10
    padding = 20
    json_path = r".\CP950\CP950-千字文.json"
    unicode_num = 1000

    crop_boxes(
        input_folder=input_folder,
        output_folder=output_folder,
        start_page=start_page,
        end_page=end_page,
        min_box_size=min_box_size,
        padding=padding,
        json_path=json_path,
        unicode_num=unicode_num,
        min_area_threshold=min_area_threshold,
    )