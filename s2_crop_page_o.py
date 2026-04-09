from PIL import Image, ImageDraw
import cv2
import numpy as np
import os
import json
import shutil


def cv_imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """支援 Windows 中文路徑的讀圖"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def cv_imwrite_unicode(path, image):
    """支援 Windows 中文路徑的寫圖"""
    try:
        ext = os.path.splitext(path)[1]
        if not ext:
            ext = ".png"
            path += ext
        success, encoded_img = cv2.imencode(ext, image)
        if not success:
            return False
        encoded_img.tofile(path)
        return True
    except Exception:
        return False


def read_json(file, unicode_num):
    with open(file, "r", encoding="utf-8-sig") as f:
        p = json.load(f)

    cp950_list = p.get("CP950", [])
    actual_num = min(unicode_num, len(cp950_list))

    unicode_list = [''] * actual_num
    for i in range(actual_num):
        raw_code = cp950_list[i].get("UNICODE", "").replace("0x", "").replace("U+", "")
        unicode_list[i] = "U+" + raw_code.upper()

    return unicode_list


def scale_adjustment(word_img, img_name):
    """調整文字大小、重心"""
    word_img = np.array(word_img)
    word_img_copy = cv2.copyMakeBorder(
        word_img, 50, 50, 50, 50,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )

    # 二值化處理
    if len(word_img_copy.shape) == 3:
        binary_word_img = cv2.cvtColor(word_img_copy, cv2.COLOR_BGR2GRAY)
    else:
        binary_word_img = word_img_copy

    binary_word_img = cv2.threshold(binary_word_img, 127, 255, cv2.THRESH_BINARY_INV)[1]

    # 如果整張圖沒有有效前景，直接回傳 resize 結果
    if cv2.countNonZero(binary_word_img) == 0:
        return cv2.resize(word_img_copy, (300, 300), interpolation=cv2.INTER_AREA)

    # 取得文字 Bounding Box
    topLeftX, topLeftY, word_w, word_h = cv2.boundingRect(binary_word_img)

    # 計算幾何中心
    cX = topLeftX + word_w // 2
    cY = topLeftY + word_h // 2

    # 標註 bounding box 和質心
    if len(word_img_copy.shape) == 2:
        annotated_img = cv2.cvtColor(word_img_copy, cv2.COLOR_GRAY2BGR)
    else:
        annotated_img = word_img_copy.copy()

    cv2.rectangle(
        annotated_img,
        (topLeftX, topLeftY),
        (topLeftX + word_w, topLeftY + word_h),
        (255, 168, 0),
        4
    )
    cv2.circle(annotated_img, (cX, cY), 10, (0, 0, 255), -1)

    # 保存標註圖片
    annotated_img_path = os.path.join("annotated_images", f"{img_name}_annotated.png")
    os.makedirs("annotated_images", exist_ok=True)
    cv_imwrite_unicode(annotated_img_path, annotated_img)

    # 數值越大文字越小，數值越小文字越大
    crop_length = 200

    h, w = word_img_copy.shape[:2]
    left_x = max(0, cX - crop_length // 2)
    right_x = min(w, cX + crop_length // 2)
    top_y = max(0, cY - crop_length // 2)
    bot_y = min(h, cY + crop_length // 2)

    final_word_img = word_img_copy[top_y:bot_y, left_x:right_x]
    return cv2.resize(final_word_img, (300, 300), interpolation=cv2.INTER_AREA)


def get_unique_filename(directory, filename):
    base, extension = os.path.splitext(filename)
    counter = 2
    unique_filename = filename

    while os.path.exists(os.path.join(directory, unique_filename)):
        unique_filename = f"{base}_{counter}{extension}"
        counter += 1

    return unique_filename


def crop_boxes(
    input_folder,
    output_folder,
    start_page,
    end_page,
    min_box_size,
    padding,
    json_path,
    unicode_num,
    min_area_threshold
):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(output_folder, exist_ok=True)

    unicode_list = read_json(json_path, unicode_num)
    if not unicode_list:
        print("錯誤：JSON 內沒有可用的 Unicode 資料。")
        return

    total_unicode_num = len(unicode_list)
    k = (start_page - 1) * 100

    print(f"起始索引 k = {k}")

    for page in range(start_page, end_page + 1):
        if k >= total_unicode_num:
            print("已處理完所有字元，提前結束。")
            break

        candidate_files = [
            f"page-{page:03d}_qr-{page}.png",
            f"page-{page}.png",
        ]

        image_path = None
        for image_file in candidate_files:
            temp_path = os.path.join(input_folder, image_file)
            if os.path.exists(temp_path):
                image_path = temp_path
                break

        print(f"處理第 {page} 頁: {image_path if image_path else candidate_files[0]}")

        if image_path is None:
            print(f"找不到圖片：{candidate_files}，跳過。")
            continue

        # Pillow 通常可處理中文路徑；保留作為保險
        try:
            image = Image.open(image_path)
        except Exception:
            print(f"PIL 無法讀取圖片：{image_path}，跳過。")
            continue

        img_np = cv_imread_unicode(image_path, cv2.IMREAD_COLOR)

        if img_np is None:
            print(f"無法讀取圖片：{image_path}，跳過。")
            continue

        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

        # 使用二值化處理，使方框更容易被檢測
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # 排除右下角 QR Code 區域
        h_img, w_img = binary.shape
        qr_size = int(min(h_img, w_img) * 0.12)
        binary[-qr_size:, -qr_size:] = 0

        # 使用輪廓檢測方框
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 依照大致行列排序
        contours = sorted(
            contours,
            key=lambda x: (cv2.boundingRect(x)[1] // 120, cv2.boundingRect(x)[0])
        )

        draw = ImageDraw.Draw(image)

        for contour in contours:
            if k >= total_unicode_num:
                break

            x, y, w, h = cv2.boundingRect(contour)

            # 排除右下角 QR Code 區域
            if x + w > img_np.shape[1] - qr_size and y + h > img_np.shape[0] - qr_size:
                continue

            # 內縮方框
            x += padding
            y += padding
            w -= 2 * padding
            h -= 2 * padding

            # 避免 padding 後變負數或太小
            if w < min_box_size or h < min_box_size:
                continue

            # 保護裁切範圍
            if x < 0 or y < 0 or x + w > img_np.shape[1] or y + h > img_np.shape[0]:
                continue

            cropped_region = img_np[y:y + h, x:x + w]
            if cropped_region.size == 0:
                continue

            cropped_image = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)

            median_filtered = cv2.medianBlur(cropped_image, 3)
            kernel = np.ones((2, 2), np.uint8)
            processed_image = cv2.morphologyEx(median_filtered, cv2.MORPH_OPEN, kernel)

            connectivity, labels, stats, centroids = cv2.connectedComponentsWithStats(
                processed_image,
                connectivity=8
            )

            for j in range(1, connectivity):
                area = stats[j, cv2.CC_STAT_AREA]
                if area < min_area_threshold:
                    processed_image[labels == j] = 0

            current_unicode = unicode_list[k]
            adjusted_image = scale_adjustment(processed_image, current_unicode)

            original_filename = f"{current_unicode}.png"
            final_filename = get_unique_filename(output_folder, original_filename)
            save_path = os.path.join(output_folder, final_filename)
            cv_imwrite_unicode(save_path, adjusted_image)

            k += 1
            cv2.rectangle(img_np, (x, y), (x + w, y + h), (255, 0, 0), 2)

        bound_output_directory = "rec_bound"
        os.makedirs(bound_output_directory, exist_ok=True)
        bound_save_path = os.path.join(bound_output_directory, f"page-{page}.png")
        cv_imwrite_unicode(bound_save_path, img_np)

    print("裁切完成。")


if __name__ == "__main__":
    input_folder = r"c:\Users\ChengYu\Desktop\ai2026s\ai2026s\hw02\02-1_crop_paper\rotated_113590051_楊承諭_千字文"
    output_folder = r"crop\crop_千字文"

    start_page = int(input("Enter start page: "))
    end_page = int(input("Enter end page: "))

    min_box_size = 180
    min_area_threshold = 10
    padding = 20
    json_path = r"./CP950/CP950-千字文.json"
    unicode_num = 78

    crop_boxes(
        input_folder,
        output_folder,
        start_page,
        end_page,
        min_box_size,
        padding,
        json_path,
        unicode_num,
        min_area_threshold
    )