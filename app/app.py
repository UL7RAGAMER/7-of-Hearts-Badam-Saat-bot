import flet as ft
import os
from app.ui import GameUI



def main(page: ft.Page):
    page.title = "Badaam Sath Bot (PDCFR+)"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 5
    GameUI(page)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

def setup_and_run():
    # 1. Force Absolute Path (Keep your previous fix)
    os.chdir(os.path.dirname(os.path.abspath(__file__))) 
    base_dir = os.path.dirname(os.path.abspath(__file__))
    upload_path = os.path.join(base_dir, "uploads")
    os.makedirs(upload_path, exist_ok=True,)
    
    # 2. SET THE SECRET KEY (The new fix)
    # This acts as a password for the upload "permission slip"
    os.environ["FLET_SECRET_KEY"] = "Siddharths_Secret_Key_123"

    print(f"Server storage ready at: {upload_path}")

    # 3. Start App
    ft.app(
        target=main, 
        view=ft.WEB_BROWSER, 
        port=8550, 
        host="0.0.0.0",
        upload_dir=upload_path 
    )