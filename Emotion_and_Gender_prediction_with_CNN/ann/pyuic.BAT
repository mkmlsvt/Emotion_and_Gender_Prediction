@echo off

echo UI CONVERTER BY BIYGRA
echo ""
echo ""
echo ""
echo ""
set /p UiName=UI Dosyasinin adi: 
echo ""
echo ""
echo ""
echo ""


call python -m PyQt5.uic.pyuic -x "%UiName%".ui -o "%UiName%".py

echo TESEKKURLER 12 NUMARA
pause