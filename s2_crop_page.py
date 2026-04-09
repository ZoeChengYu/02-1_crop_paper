from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


OUTPUT_SIZE = 300
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def read_unicode_list(json_path, unicode_num):
    """
    讀取 CP950 JSON，轉成 U+XXXX / U+XXXXX 格式清單
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cp950 = data["CP950"][:unicode_num]
    return [item["UNICODE"].replace("0x", "U+").upper() for item in cp950]


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


def extract_page_number(path):
    """
    支援：
    - page-001_qr-1.png
    - page-001.png
    - page-1.png
    """
    name = Path(path).name
    patterns = [
        r"page-(\d+)_qr-(\d+)\.png$",
        r"page-(\d+)\.png$",
    ]
    for pattern in patterns:
        m = re.match(pattern, name, re.IGNORECASE)
        if m:
            return int(m.group(1))
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
    """
    找整頁的方框，每頁預期 100 格
    """
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)[1]

    img_h, img_w = binary.shape
    qr_size = int(min(img_h, img_w) * QR_RATIO)
    binary[-qr_size:, -qr_size:] = 0

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    debug_dir = Path("debug_binary")
    debug_dir.mkdir(parents=True, exist_ok=True)
    imwrite_unicode(debug_dir / f"{debug_name}_binary.png", binary)

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w < 110 or h < 110:
            continue

        ratio = w / float(h)
        if 0.75 <= ratio <= 1.25:
            boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: (round(b[1] / 120), b[0]))
    return boxes[:100]


def draw_detected_boxes(img_color, boxes, output_path):
    preview = img_color.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 0, 0), 2)
    imwrite_unicode(output_path, preview)


def clean_small_noise(gray_img, min_area_threshold=10):
    """
    黑字白底灰階圖 -> 清掉太小的黑色雜訊
    """
    inv = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY_INV)[1]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)

    cleaned_inv = inv.copy()
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area_threshold:
            cleaned_inv[labels == i] = 0

    cleaned = 255 - cleaned_inv
    return cleaned


def preprocess_cropped_image(cropped_gray, min_area_threshold):
    median_filtered = cv2.medianBlur(cropped_gray, 3)
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(median_filtered, cv2.MORPH_OPEN, kernel)
    cleaned = clean_small_noise(opened, min_area_threshold)
    return cleaned

"""
def erase_border_lines(gray_img, border=12):
    """'''
    只移除像稿紙框線的細長邊界元件，
    避免把真的字刪掉。
    '''"""
    result = gray_img.copy()

    binary_inv = cv2.threshold(result, 200, 255, cv2.THRESH_BINARY_INV)[1]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_inv, connectivity=8)

    h, w = binary_inv.shape
    cleaned_inv = binary_inv.copy()

    for lab in range(1, num_labels):
        x = stats[lab, cv2.CC_STAT_LEFT]
        y = stats[lab, cv2.CC_STAT_TOP]
        ww = stats[lab, cv2.CC_STAT_WIDTH]
        hh = stats[lab, cv2.CC_STAT_HEIGHT]
        area = stats[lab, cv2.CC_STAT_AREA]

        touches_top = y <= border
        touches_bottom = (y + hh) >= (h - border)
        touches_left = x <= border
        touches_right = (x + ww) >= (w - border)

        is_top_horizontal_line = touches_top and hh <= 12 and ww >= int(w * 0.45)
        is_bottom_horizontal_line = touches_bottom and hh <= 12 and ww >= int(w * 0.45)

        is_left_vertical_line = touches_left and ww <= 12 and hh >= int(h * 0.45)
        is_right_vertical_line = touches_right and ww <= 12 and hh >= int(h * 0.45)

        looks_like_border = (
            is_top_horizontal_line
            or is_bottom_horizontal_line
            or is_left_vertical_line
            or is_right_vertical_line
        ) and area <= int(h * w * 0.12)

        if looks_like_border:
            cleaned_inv[labels == lab] = 0

    return 255 - cleaned_inv
    """


def normalize_character_image(word_img, img_name, save_annotated=True, output_size=300):
    if len(word_img.shape) == 3:
        gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = word_img.copy()

    gray = cv2.copyMakeBorder(
        gray,
        20, 20, 20, 20,
        borderType=cv2.BORDER_CONSTANT,
        value=255
    )

    binary_inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]

    if cv2.countNonZero(binary_inv) == 0:
        return np.full((output_size, output_size), 255, dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(binary_inv)

    if save_annotated:
        annotated_dir = Path("annotated_images")
        annotated_dir.mkdir(parents=True, exist_ok=True)

        annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 168, 0), 2)
        imwrite_unicode(annotated_dir / f"{img_name}_annotated.png", annotated)

    tight = gray[y:y + h, x:x + w]

    pad_x = max(12, int(w * 0.16))
    pad_y = max(12, int(h * 0.16))

    padded = cv2.copyMakeBorder(
        tight,
        pad_y, pad_y, pad_x, pad_x,
        borderType=cv2.BORDER_CONSTANT,
        value=255,
    )

    safety = 12
    padded = cv2.copyMakeBorder(
        padded,
        safety, safety, safety, safety,
        borderType=cv2.BORDER_CONSTANT,
        value=255,
    )

    ph, pw = padded.shape[:2]
    side = max(ph, pw)

    square = np.full((side, side), 255, dtype=np.uint8)
    offset_y = (side - ph) // 2
    offset_x = (side - pw) // 2
    square[offset_y:offset_y + ph, offset_x:offset_x + pw] = padded

    result = cv2.resize(square, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return result


def crop_boxes(
    input_folder,
    output_folder,
    start_page,
    end_page,
    min_box_size,
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

        if len(boxes) != 100:
            print(f"  Warning: 預期 100 格，實際偵測 {len(boxes)} 格")

        draw_detected_boxes(
            img_color,
            boxes,
            bound_output_directory / f"page-{page_num}.png"
        )

        page_saved = 0

        for (x, y, w, h) in boxes:
            if k >= unicode_num:
                break

            # 小幅內縮
            pad_left = 8
            pad_right = 8
            pad_top = 10
            pad_bottom = 8

            x2 = x + pad_left
            y2 = y + pad_top
            w2 = w - pad_left - pad_right
            h2 = h - pad_top - pad_bottom

            if w2 <= 0 or h2 <= 0:
                continue
            if w2 < min_box_size or h2 < min_box_size:
                continue

            cropped = img_color[y2:y2 + h2, x2:x2 + w2]
            if cropped.size == 0:
                continue

            cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            processed = preprocess_cropped_image(cropped_gray, min_area_threshold)

            # 不做 erase_border_lines，避免把貼邊筆畫誤刪
            final_img = normalize_character_image(
                processed,
                unicode_list[k],
                save_annotated=True,
                output_size=OUTPUT_SIZE,
            )

            filename = make_unique_filename(unicode_list[k], used_names)
            out_path = output_folder / filename
            ok = imwrite_unicode(out_path, final_img)

            if ok:
                total_saved += 1
                page_saved += 1
                k += 1
            else:
                print(f"  write failed: {out_path}")

        print(f"  saved on this page: {page_saved}")

    print(f"\n完成裁切，總共輸出 {total_saved} 張字圖")
    print(f"輸出資料夾：{output_folder.resolve()}")


if __name__ == "__main__":
    input_folder = r"c:\Users\ChengYu\Desktop\ai2026s\ai2026s\hw02\02-1_crop_paper\rotated_113590051_楊承諭_標點符號"
    output_folder = r"crop\crop_標點符號"

    start_page = int(input("Enter start page: "))
    end_page = int(input("Enter end page: "))

    min_box_size = 120
    min_area_threshold = 10
    json_path = r".\CP950\CP950-標點符號.json"
    unicode_num = 356

    crop_boxes(
        input_folder=input_folder,
        output_folder=output_folder,
        start_page=start_page,
        end_page=end_page,
        min_box_size=min_box_size,
        json_path=json_path,
        unicode_num=unicode_num,
        min_area_threshold=min_area_threshold,
    )