@rem Install python 3.10
python -m venv venv
call .\venv\Scripts\activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-with-fixes-for-windows-and-for-the-bat-file.txt
git clone https://github.com/Atharva-Phatak/torchflare
cd torchflare
pip install -r requirements.txt
call python setup install