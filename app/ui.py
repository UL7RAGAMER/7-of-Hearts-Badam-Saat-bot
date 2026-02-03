import os
from app.bot import Bot
from app.gamestate import GameState
from app.scanner import CardScanner
import flet as ft

from utils.constant import SUITS, SUIT_SYMBOL, RANKS, CARD_BG, SUIT_COLORS
from utils.helper_functions import card_parts
class GameUI:
    def __init__(self, page: ft.Page):
            self.page = page
            self.state = GameState()
            self.bot = Bot()
            self.scanner = CardScanner('cards.pt')
            
            # --- 1. SETUP FILE PICKER with UPLOAD Handler ---
            # IMPORTANT: 'on_upload' is triggered when the file finishes uploading to PC
            self.file_picker = ft.FilePicker(
                on_result=self._on_file_picked,
                on_upload=self._on_upload_complete
            )
            self.page.overlay.append(self.file_picker)
            self.page.update()

            self.board_row = ft.Row(
                scroll=ft.ScrollMode.ALWAYS, 
                vertical_alignment=ft.CrossAxisAlignment.START,
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=15
            )
            
            self.hand_row = ft.Row(
                scroll=ft.ScrollMode.ALWAYS, 
                height=100,
                alignment=ft.MainAxisAlignment.CENTER
            )
            self.info = ft.Text("Select hand (13 cards)", weight="bold", size=14)
            self._build_setup()

    def _build_setup(self):
        self.setup_rows = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True)
        self.count = ft.Text("Selected: 0/13", weight="bold")

        for r_idx in range(13): 
            row_controls = []
            for s_idx in range(4):
                cid = s_idx * 13 + r_idx
                rank_str, suit_str = RANKS[r_idx], SUITS[s_idx]
                is_selected = cid in self.state.my_hand
                
                card_btn = ft.Container(
                    content=ft.Row([
                        ft.Text(rank_str, weight="bold", size=14, color=SUIT_COLORS[suit_str]),
                        ft.Text(SUIT_SYMBOL[suit_str], size=16, color=SUIT_COLORS[suit_str]),
                    ], alignment=ft.MainAxisAlignment.CENTER, spacing=5),
                    bgcolor=ft.Colors.BLUE_100 if is_selected else CARD_BG,
                    border=ft.border.all(1, ft.Colors.GREY_300),
                    border_radius=6,
                    alignment=ft.alignment.center,
                    data=cid,
                    on_click=self._toggle,
                    height=45,
                    expand=1
                )
                row_controls.append(card_btn)
            self.setup_rows.controls.append(ft.Row(row_controls, spacing=5))
        
        self.scan_btn = ft.ElevatedButton(
            "📷 Cam/Upload", 
            icon=ft.Icons.CAMERA_ALT, 
            bgcolor=ft.Colors.PURPLE_700,
            color=ft.Colors.WHITE,
            on_click=lambda _: self.file_picker.pick_files(
                allow_multiple=False,
                file_type=ft.FilePickerFileType.IMAGE
            )
        )
        self.page.add(
            ft.Column([
                ft.Row([ft.Text("Select Hand", size=18, weight="bold"), self.count], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Divider(height=10, thickness=1),
                ft.Row([self.scan_btn], alignment=ft.MainAxisAlignment.END), # Added Button Here
                ft.Container(self.setup_rows, expand=True), 
                ft.ElevatedButton("Start Game", on_click=self._start, width=float("inf")), 
            ], expand=True)
        )

    # --- UPLOAD & SCAN LOGIC ---
    def _on_file_picked(self, e: ft.FilePickerResultEvent):
        """Called when user selects a file on their phone."""
        if not e.files: return
        
        upload_list = []
        for f in e.files:
            # 1. Generate the secure upload URL for the 'uploads' folder
            # The '600' is the URL validity duration in seconds
            url = self.page.get_upload_url(f.name, 600)
            
            # 2. Add to the list with the URL attached
            upload_list.append(
                ft.FilePickerUploadFile(f.name, upload_url=url)
            )

        # 3. Trigger the actual upload using the URLs
        self.file_picker.upload(upload_list)
        
        self.count.value = "Uploading..."
        self.count.update()

    def _on_upload_complete(self, e: ft.FilePickerUploadEvent):
        """Called when the file actually arrives on the PC."""
        if e.error:
            print(f"--- UPLOAD FAILED ---")
            print(f"Error: {e.error}")
            self.count.value = "Upload Error"
            self.count.update()
            return

        
        file_name = e.file_name
        # DEBUG: Print exactly where we are looking
        base_dir = os.path.dirname(os.path.abspath(__file__))
        target_folder = os.path.join(base_dir, "uploads")
        file_path = os.path.join(target_folder, file_name)
        
        print(f"--- UPLOAD DEBUG ---")
        print(f"Looking in: {target_folder}")
        print(f"For file: {file_name}")
        print(f"Folder contents: {os.listdir(target_folder)}")
        # 1. Manually build the full path. 
        # Flet puts it in the 'uploads' folder next to this script.
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "uploads", file_name)
        
        print(f"Looking for file at: {file_path}")
        
        # 2. Safety Wait (sometimes disk write lags by 100ms)
        import time
        if not os.path.exists(file_path):
             time.sleep(0.5) 
        
        if not os.path.exists(file_path):
             print(f"Error: File missing. Folder contents: {os.listdir(os.path.join(base_dir, 'uploads'))}")
             self.count.value = "Upload Failed"
             self.count.update()
             return

        try:
            detected_ids = self.scanner.detect_from_file(file_path)
            if detected_ids:
                for cid in detected_ids:
                    if len(self.state.my_hand) < 13:
                        self.state.my_hand.add(cid)
                
                self._update_setup_buttons()
                self.count.value = f"Selected: {len(self.state.my_hand)}/13"
                self.count.update()
            else:
                self.count.value = "No cards detected."
                self.count.update()
        except Exception as ex:
            print(f"Scan Error: {ex}")
            self.count.value = "Scan Failed"
            self.count.update()

    def _update_setup_buttons(self):
        """Refreshes button colors after scanning."""
        for row in self.setup_rows.controls:
            for container in row.controls:
                cid = container.data
                if cid in self.state.my_hand:
                    container.bgcolor = ft.Colors.BLUE_100
                else:
                    container.bgcolor = CARD_BG
        self.setup_rows.update()

    def _toggle(self, e):
        cid = e.control.data
        if cid in self.state.my_hand:
            self.state.my_hand.remove(cid)
            e.control.bgcolor = CARD_BG
        elif len(self.state.my_hand) < 13:
            self.state.my_hand.add(cid)
            e.control.bgcolor = ft.Colors.BLUE_100
        
        self.count.value = f"Selected: {len(self.state.my_hand)}/13"
        e.control.update() 
        self.count.update()

    def _start(self, _):
        self.page.controls.clear() 
        self.page.add(self._game_view())
        self.refresh()

    def _game_view(self):
            return ft.Column([
                ft.Container(
                    content=ft.Row([
                        ft.Text("OPPONENT MOVES", weight="bold", size=12, color=ft.Colors.GREY_500),
                        ft.Row([
                            ft.IconButton(icon=ft.Icons.UNDO, icon_color=ft.Colors.RED_400, on_click=self._click_undo, icon_size=20),
                            ft.Container(self.info, alignment=ft.alignment.center_right),
                        ], spacing=5)
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    padding=10
                ),
                
                ft.Container(
                    content=ft.Column(
                        [self.board_row], 
                        scroll=ft.ScrollMode.AUTO,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER 
                    ),
                    expand=True,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.WHITE),
                    border_radius=10,
                ), 
                
                ft.Divider(height=1, thickness=1),
                
                ft.Container(
                    content=ft.Column([
                        ft.Text("YOUR HAND", weight="bold", size=14),
                        self.hand_row,
                    ], 
                    spacing=5,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER
                    ),
                    padding=ft.padding.only(left=10, right=10, bottom=20, top=10),
                ),
            ], expand=True)

    def refresh(self):
        self.board_row.controls.clear()

        for suit_idx in range(4):
            suit_stack = []
            suit_stack.append(
                ft.Container(
                    ft.Text(SUIT_SYMBOL[SUITS[suit_idx]], size=20, weight="bold", color=SUIT_COLORS[SUITS[suit_idx]]),
                    alignment=ft.alignment.center,
                    height=30
                )
            )

            low, high = self.state.table_bounds[suit_idx] 
            
            for r_idx in range(13):
                cid = suit_idx * 13 + r_idx
                rank_val = r_idx + 1
                is_played = cid in self.state.played_cards
                is_playable = False
                
                if not is_played:
                    if low is None:
                        if rank_val == 7: is_playable = True
                    else:
                        if rank_val == low - 1 or rank_val == high + 1: is_playable = True

                if is_played:
                    suit_stack.append(self._played_mini(cid))
                elif is_playable:
                    suit_stack.append(self._ghost_mini(cid))
                else:
                    suit_stack.append(self._empty_mini())

            self.board_row.controls.append(
                ft.Column(suit_stack, spacing=4, horizontal_alignment=ft.CrossAxisAlignment.CENTER)
            )

        self._render_hand()
        
        # BOT SUGGESTION
        suggestion = self.bot.move(self.state)
        
        if suggestion is not None:
            r, s = card_parts(suggestion)
            self.info.value = f"Bot: {r}{SUIT_SYMBOL[s]}"
            self.info.color = ft.Colors.GREEN
        else:
            self.info.value = "Pass / No Moves"
            self.info.color = ft.Colors.RED
        self.info.update()
        self.page.update()

    def _play(self, cid, is_user):
        self.state.play_card(cid, user=is_user)
        self.refresh()

    def _click_undo(self, e):
        self.state.undo_last_move()
        self.refresh()

    def _played_mini(self, cid):
        r, _ = card_parts(cid)
        suit_idx, _ = divmod(cid, 13)
        return ft.Container(
            content=ft.Text(r, size=11, weight="bold", color=ft.Colors.WHITE),
            bgcolor=SUIT_COLORS[SUITS[suit_idx]],
            width=35, height=24,
            border_radius=4,
            alignment=ft.alignment.center
        )

    def _ghost_mini(self, cid):
        return ft.Container(
            content=ft.Text("+", size=14, color=ft.Colors.BLACK, weight="bold"),
            bgcolor=ft.Colors.GREEN_400, 
            width=35, height=24,
            border_radius=4,
            alignment=ft.alignment.center,
            on_click=lambda _, c=cid: self._play(c, is_user=False)
        )

    def _empty_mini(self):
        return ft.Container(
            bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.GREY_400),
            width=35, height=24,
            border_radius=2
        )

    def _render_hand(self):
        self.hand_row.controls.clear()
        
        suggestion = self.bot.move(self.state)

        for cid in sorted(self.state.my_hand):
            r, s = card_parts(cid)
            is_suggested = (cid == suggestion)
            
            self.hand_row.controls.append(
                ft.Container(
                    content=ft.Column([
                        ft.Text(r, weight="bold", color=SUIT_COLORS[s], size=14),
                        ft.Text(SUIT_SYMBOL[s], color=SUIT_COLORS[s], size=12),
                    ], alignment=ft.MainAxisAlignment.CENTER, spacing=0),
                    width=45, height=65,
                    bgcolor=CARD_BG,
                    border=ft.border.all(3 if is_suggested else 1, ft.Colors.GREEN if is_suggested else ft.Colors.GREY_400),
                    border_radius=6,
                    alignment=ft.alignment.center,
                    on_click=lambda _, c=cid: self._play(c, is_user=True),
                )
            )
