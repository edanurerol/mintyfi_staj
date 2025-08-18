@echo off
setlocal enabledelayedexpansion

:: 1) Nutuk belgelerini ingest et
echo [1/2] Nutuk belgeleri ingest ediliyor...
curl -s -X POST http://127.0.0.1:8000/ingest ^
  -H "Content-Type: application/json" ^
  -d "{\"dir_path\":\"docs\"}"
echo.
echo.

:: 2) Kullanıcıdan soru al
set /p QUESTION=Soru girin: 

:: 3) Soruyu gönder ve cevabı al
echo [2/2] Sorunuz: !QUESTION!
echo.
curl -s -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"!QUESTION!\",\"top_k\":4}" > response.json

:: 4) JSON içinden sadece cevabı göster (PowerShell ile)
echo.
echo ---- CEVAP ----
powershell -Command "(Get-Content response.json | ConvertFrom-Json).answer"
echo.
pause
