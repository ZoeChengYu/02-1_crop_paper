import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from natsort import ns, natsorted


def parse_args():
    parser = argparse.ArgumentParser(description='Step 1 rotate scan page')

    parser.add_argument(
        '--name',
        help='To rotated page from target folder',
        default=None,
        type=str
    )

    args = parser.parse_args()

    if args.name is None:
        args.name = input("請輸入目標資料夾名稱：").strip()

    return args


def boxSize(arr):
    """獲取 bbox 的最大以及最小 X, Y

    Keyword arguments:
        arr -- bbox
    Return:
        (min_x, min_y, max_x, max_y)
    """
    box_roll = np.rollaxis(arr, 1, 0)
    xmax = int(np.amax(box_roll[0]))
    xmin = int(np.amin(box_roll[0]))
    ymax = int(np.amax(box_roll[1]))
    ymin = int(np.amin(box_roll[1]))
    return (xmin, ymin, xmax, ymax)


def get_skew_angle(image) -> float:
    """用整張圖的長直線估計傾斜角度（單位：度）"""
    if image is None or image.size == 0:
        return 0.0

    # 邊緣
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # 找線段
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=min(image.shape[:2]) // 4,
        maxLineGap=20
    )

    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue

        angle = np.degrees(np.arctan2(dy, dx))

        # 只保留接近水平的線，避免垂直邊干擾
        if -45 < angle < 45:
            angles.append(angle)

    if not angles:
        return 0.0

    return float(np.median(angles))


def saveImage(image, now_page, index):
    global result_path
    cv2.imwrite(f'./{result_path}/page-{index + 1:03d}_qr-{now_page}.png', image)


def try_decode_with_variants(image):
    """對同一張灰階圖做多種前處理後依序偵測 QR code"""
    if image is None or image.size == 0:
        return None, None

    qrcode = cv2.QRCodeDetector()
    h, w = image.shape[:2]

    candidates = []

    # 原圖
    candidates.append(("orig", image))

    # OTSU 二值化
    th1 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    candidates.append(("otsu", th1))

    # 反相後 OTSU
    inv = 255 - image
    th2 = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    candidates.append(("inv_otsu", th2))

    # 輕微模糊
    blur3 = cv2.GaussianBlur(image, (3, 3), 0)
    candidates.append(("blur3", blur3))

    # 放大 2 倍
    up2 = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    candidates.append(("up2", up2))

    # 放大 2 倍後 OTSU
    up2_th = cv2.threshold(up2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    candidates.append(("up2_otsu", up2_th))

    for _, img_try in candidates:
        now_page, bbox, _ = qrcode.detectAndDecode(img_try)
        if bbox is not None:
            hh, ww = img_try.shape[:2]
            if hh != h or ww != w:
                scale_x = w / ww
                scale_y = h / hh
                bbox = bbox.copy()
                bbox[:, :, 0] *= scale_x
                bbox[:, :, 1] *= scale_y
            return now_page, bbox

    return None, None


def qrcode_finder(image):
    """搜尋 QR Code 位置，失敗時固定回傳 (None, None)"""
    now_page, bbox = try_decode_with_variants(image)
    return now_page, bbox


def crop_region(image, region_name, center_h, center_w, height, width):
    """依區域名稱裁切影像"""
    if region_name == "full":
        return image, 0, 0

    if region_name == "left_top":
        return image[0:center_h, 0:center_w], 0, 0

    if region_name == "right_top":
        return image[0:center_h, center_w:width], center_w, 0

    if region_name == "left_bottom":
        return image[center_h:height, 0:center_w], 0, center_h

    if region_name == "right_bottom":
        return image[center_h:height, center_w:width], center_w, center_h

    return image, 0, 0


def get_qrcode_crop(gray, bbox, region, x_offset, y_offset, width, height, scale=30):
    """依 bbox 與區域偏移，從原圖灰階圖中取出 QR code 區塊"""
    box = boxSize(bbox[0])

    x1 = max(box[0] + x_offset - scale, 0)
    y1 = max(box[1] + y_offset - scale, 0)
    x2 = min(box[2] + x_offset + scale, width)
    y2 = min(box[3] + y_offset + scale, height)

    crop = gray[y1:y2, x1:x2]
    return crop


def rotate_img(file_path, index) -> bool:
    """主程式，以 QR Code 旋轉稿紙"""
    try:
        img = cv2.imread(file_path)
        if img is None:
            print(f"\n錯誤檔案：{file_path}，cv2.imread 失敗")
            return False
    except Exception as e:
        print(f"\n錯誤檔案：{file_path}, {e}")
        return False

    height, width, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 不要一開始做太重的模糊，避免 QR 細節流失
    work = gray

    center_w = width // 2
    center_h = height // 2

    search_order = [
        "full",
        "right_bottom",
        "left_top",
        "right_top",
        "left_bottom",
    ]

    now_page = None
    bbox = None
    used_region = None
    x_offset = 0
    y_offset = 0

    for region in search_order:
        region_img, x_off, y_off = crop_region(
            work, region, center_h, center_w, height, width
        )
        candidate_now_page, candidate_bbox = qrcode_finder(region_img)

        if candidate_bbox is not None:
            now_page = candidate_now_page
            bbox = candidate_bbox
            used_region = region
            x_offset = x_off
            y_offset = y_off
            break

    if bbox is None:
        print(f"找不到 QRCode bbox：{file_path}")
        return False

    if now_page == 'https://tjhsieh.github.io/c/ct/ct2023s/syllabus/index.html':
        now_page = str(index + 1)
    elif now_page == '' or now_page is None:
        now_page = str(index + 1)

    qrcode_crop = get_qrcode_crop(
        gray, bbox, used_region, x_offset, y_offset, width, height, scale=30
    )

    if qrcode_crop is None or qrcode_crop.size == 0:
        print(f"QRCode 區域裁切失敗：{file_path}")
        return False

    angle = get_skew_angle(qrcode_crop)

    M = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        img,
        M,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    saveImage(rotated, now_page, index)
    return True


if __name__ == '__main__':
    args = parse_args()
    target_folder = args.name

    target_path = f'./{target_folder}'
    result_path = f'rotated_{target_folder}'

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    print(f"Handling page rotation, student id = {target_folder}")

    errorList = []
    allFileList = os.listdir(target_path)
    allFileList = natsorted(allFileList, alg=ns.PATH)

    page_count = {}

    for index in tqdm(range(len(allFileList))):
        filePath = target_path + "/" + allFileList[index]
        if not rotate_img(filePath, index):
            errorList.append(allFileList[index])

    print("Rotate successfully")

    if len(errorList):
        print("The following is the wrong file, please rotate it yourself：")
        for errPath in errorList:
            print(errPath)