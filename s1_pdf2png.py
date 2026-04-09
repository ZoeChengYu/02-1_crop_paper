from pdf2image import convert_from_path
from tqdm import tqdm
import os

# 設定 Poppler 路徑 (請確認你的 Poppler 安裝位置)
poppler_path = r"C:\Users\ChengYu\anaconda3\envs\fontenv\Library\bin" 

# 轉換 PDF 為 PNG 圖片
def pdf_to_png(pdf_path, output_folder, start_page=1, end_page=None, dpi=600):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("讀取 PDF 文件中，請稍等…")

    try:
        # 轉換 PDF 頁面為圖片
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=start_page,
            last_page=end_page,
            poppler_path=poppler_path
        )

        # 進度條顯示
        for page, img in tqdm(enumerate(images, start=start_page), total=len(images), desc="轉換進度"):
            img.save(os.path.join(output_folder, f"page-{page}.png"), "PNG")

        print(f"轉換完成！圖片存儲於：{output_folder}")

    except Exception as e:
        print(f"發生錯誤：{e}")

# 使用者輸入
pdf_path = input("請輸入 PDF 文件的完整路徑（包含 .pdf）：").strip('"')
output_folder = input("請輸入輸出圖片的資料夾（設定為rotated_你的學號）：").strip()
start_page = int(input("請輸入要切割的 PDF 起始頁：").strip())
end_page = int(input("請輸入要切割的 PDF 結束頁：").strip())

pdf_to_png(pdf_path, output_folder, start_page, end_page)
