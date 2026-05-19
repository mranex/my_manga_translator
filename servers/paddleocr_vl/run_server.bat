@echo off
cd /d "C:\Nghich\Manga-Translator\tools\llama.cpp"
call C:\Nghich\Manga-Translator\tools\llama.cpp\llama-server.exe ^
  -m ^
  C:\Nghich\Manga-Translator\model\paddleocr_vl\model.gguf ^
  --mmproj ^
  C:\Nghich\Manga-Translator\model\paddleocr_vl\mmproj.gguf ^
  --host ^
  127.0.0.1 ^
  --port ^
  8080 ^
  -c ^
  8192 ^
  -ngl ^
  99 ^
  --temp ^
  0
echo.
echo Server stopped. Press any key to close this window.
pause >nul
