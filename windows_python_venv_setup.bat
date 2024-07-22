@echo off
(
echo Installing requirements.txt to venv folder via pip... 
echo --- CHECK PROGRESS AT venv_setup.log ---
echo Do not close this window until you are asked to Press any key to continue . . .

(
mkdir venv
python -m venv venv
"%CD%\venv\Scripts\activate.bat"&&"%CD%\venv\Scripts\python.exe" -m pip install --upgrade pip&"%CD%\venv\Scripts\python.exe" -m pip install --upgrade pip&&"%CD%\venv\Scripts\python.exe" -m pip install -r requirements.txt&&"%CD%\venv\Scripts\deactivate.bat"

echo check progress at venv_setup.log ...
) > venv_setup.log 2>&1 )&&@pause

