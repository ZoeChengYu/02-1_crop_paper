@echo off
setlocal enabledelayedexpansion
set "POPPLER_BIN=C:\Users\ChengYu\anaconda3\envs\fontenv\Library\bin\pdftoppm.exe"

for %%f in (*.pdf) do (
    echo [*] Processing... %%f

    :: 抓取檔名並建立資料夾
    set "outdir=rotated_jpg_%%~nf"
    mkdir "!outdir!" 2>nul
    
    :: 執行轉換
    "!POPPLER_BIN!" -png -r 300 "%%f" "!outdir!\page"
    
    :: --- 去除補零 ---
    pushd "!outdir!"
    for %%i in (page-*.png) do (
        set "oldname=%%i"
        :: 抓取 "-" 之後的數字部分
        for /f "tokens=2 delims=-" %%a in ("%%~ni") do (
            set "num=%%a"
            :: 移除數字開頭的 0
            set /a "plain_num=1!num! %% 1000"
            set /a "plain_num=!plain_num! - 1000"
            :: 重新命名為不補零的格式
            for /f "tokens=* delims=0" %%b in ("!num!") do (
                if "%%b"=="" (set "final_num=0") else (set "final_num=%%b")
                if not "!oldname!"=="page-!final_num!.jpg" (
                    ren "!oldname!" "page-!final_num!.jpg"
                )
            )
        )
    )
    popd
)

echo.
echo === Conversion Completed ===
