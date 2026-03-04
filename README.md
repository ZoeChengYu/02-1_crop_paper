# Crop Paper
```
# [稿紙對應編號]
1. 千字文
2. 長恨歌
3. 洛神賦
4. 詩經
5. 部首、注音、日文
6. 英文、數字
7. 標點符號
8. 姓氏名字號
```
## 🔆環境建置(初次設定)
### 1. 下載專案
點擊右上角綠色 `CODE` --> `DOWNLOAD ZIP`
### 2. 安裝 Conda
點擊 [Anaconda](https://www.anaconda.com/download) 註冊帳號，並安裝 miniconda
### 3. 建立虛擬環境
```
conda create --name fontenv python=3.8
```
### 4. 啟動虛擬環境
```
conda activate fontenv
```
### 5. 切換目錄至 `02-1_crop_paper`
```
# 請根據下載的專案資料夾位置更改路徑
cd D:\NTUT\AI\Font-Project\02-1_crop_paper
```
### 6. 安裝套件
```
pip install -r requirements.txt
```
### 7. 電子書寫需額外下載
```
pip install pdf2image
conda install -c conda-forge poppler
```
-----
## 🔆啟動與進入專案
### 1. 啟動虛擬環境
```
conda activate fontenv
```
### 2. 移動至專案目錄
```
# 請根據下載的專案資料夾位置更改路徑
cd D:\NTUT\AI\Font-Project\02-1_crop_paper
```

-----
## 執行專案
詳情請前往以下連結查看簡報
https://docs.google.com/presentation/d/16fZ1fT2xa7GJ0AxGAJZxEh4rVeCELIad/edit?usp=sharing&ouid=116834695962505686161&rtpof=true&sd=true

### 紙本裁切
#### 0. 資料準備
將寫完字的掃描檔案放在以學號命名文件夾底下：{學號}_{稿紙標題} \ 每一張掃描檔檔名.jpg

#### 1. 旋轉校正
```
# 擇一執行
python –name {資料夾名稱} s1_rotate_page.py
python s1_rotate_page.py
```
結果儲存於 ./rotated_{學號}_{稿紙標題}

<span id="crop_paper"></span>
#### 2. 切割稿紙
- 將 ` \01-1_generate_paper\2_generate_manuscript\CP950` 複製到 `\02-1_crop_paper`
- 在 `s2_crop_page.py` 更改 `image_folder`, `json_path` 的位置
  - `image_folder`：校正後的資料夾位置
  - `json_path`：原先製作此稿紙的 `CP950-{標題}` 位置
- 執行程式
    ```
    python s2_crop_page.py
    ```
    - 輸入要求
      - 開始切割頁面
      - 結束頁
---
### 電子檔裁切
#### 0. 資料準備
- 將寫好的稿紙(.pdf) 放在 02-1_crop_paper 底下
- 找到安裝 `poppler` 的位置
  - 參考位置：`C:\Users\Users\.conda\envs\fontenv\Library\bin`

#### 1. 轉檔校正
##### 1-a. 使用 pdf2image
```
python s1_pdf2png.py
```
- 請根據需求輸入
  - pdf 檔的位置
  - 圖片要存的資料夾
  - 需要的起始跟結束頁碼

##### 1-b. 使用 batch file
請先檢查 `pdf2png.bat` 中 `pdftoppm.exe` 路徑是否正確
```
1. 檔案位置 /02-1_crop_paper/rotation/
2. 將要轉檔的稿紙(.pdf) 放至 rotation 底下
3. 按兩下 pdf2png.bat
```
若執行失敗，請將 `pdf2png.bat` 重新編碼為 `ANSI`

#### 2. 切割稿紙
與 [紙本裁切](#crop_paper) 相同