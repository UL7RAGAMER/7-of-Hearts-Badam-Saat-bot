import flet as ft
import torch
import sys
import os

# Graceful import for the Agent
try:
    from agent.dqn import DQNAgent
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    print("Warning: 'agent.dqn' not found. Bot will play randomly or pass.")

# ======================
# CONSTANTS & UTILITIES
# ======================
SUITS = ["Hearts", "Diamonds", "Clubs", "Spades"]
SUIT_SYMBOL = {"Hearts": "♥", "Diamonds": "♦", "Clubs": "♣", "Spades": "♠"}
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

SUIT_COLORS = {
    "Hearts": ft.Colors.RED_600,
    "Diamonds": ft.Colors.RED_600,
    "Clubs": ft.Colors.BLACK,
    "Spades": ft.Colors.BLACK,
}

CARD_BG = ft.Colors.WHITE
SEVEN_RANK = 6

def card_parts(card_id):
    suit, rank = divmod(card_id, 13)
    return RANKS[rank], SUITS[suit]

# ======================
# GAME STATE (LOGIC)
# ======================
class GameState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.my_hand = set()
        self.played_cards = set()
        self.table_bounds = {i: [None, None] for i in range(4)}
        self.history = []
        self.current_turn = 0 

    def play_card(self, cid, user=False):
        suit, rank = divmod(cid, 13)
        prev_bounds = list(self.table_bounds[suit])
        
        self.history.append({
            "cid": cid,
            "user": user,
            "suit": suit,
            "prev_bounds": prev_bounds
        })

        low, high = self.table_bounds[suit]
        if low is None:
            self.table_bounds[suit] = [rank, rank]
        else:
            self.table_bounds[suit][0] = min(low, rank)
            self.table_bounds[suit][1] = max(high, rank)

        self.played_cards.add(cid)
        if user and cid in self.my_hand:
            self.my_hand.remove(cid)

        self.current_turn = (self.current_turn + 1) % 4

    def undo_last_move(self):
        if not self.history: return
        last_move = self.history.pop()
        if last_move['cid'] in self.played_cards:
            self.played_cards.remove(last_move['cid'])
        self.table_bounds[last_move['suit']] = last_move['prev_bounds']
        if last_move['user']:
            self.my_hand.add(last_move['cid'])
        self.current_turn = (self.current_turn - 1) % 4

    def legal_moves(self, hand):
        moves = []
        for cid in hand:
            suit, rank = divmod(cid, 13)
            low, high = self.table_bounds[suit]
            if rank == SEVEN_RANK:
                moves.append(cid)
            elif low is not None and (rank == low - 1 or rank == high + 1):
                moves.append(cid)
        return moves

    def to_tensor(self, device):
        # 1. Player's Hand (52)
        hand = torch.zeros(52, device=device)
        if self.my_hand: hand[list(self.my_hand)] = 1.0
        
        # 2. Played Cards (52)
        played = torch.zeros(52, device=device)
        if self.played_cards: played[list(self.played_cards)] = 1.0
        
        # 3. Table Bounds (8)
        bounds = torch.full((8,), -1.0, device=device)
        for s in range(4):
            low, high = self.table_bounds[s]
            if low is not None: bounds[s * 2] = (low - 6) / 6.0
            if high is not None: bounds[s * 2 + 1] = (high - 6) / 6.0
            
        # 4. Opponent Info (3)
        opponents = torch.tensor([0.3, 0.3, 0.3], device=device)
        
        # 5. Current Turn One-Hot Encoding (4) - MISSING PIECE
        turn_one_hot = torch.zeros(4, device=device)
        turn_one_hot[self.current_turn] = 1.0
        
        # Total: 52 + 52 + 8 + 3 + 4 = 119
        return torch.cat([hand, played, bounds, opponents, turn_one_hot])

# ======================
# BOT
# ======================
class Bot:
    def __init__(self):
        self.ok = False
        if not AGENT_AVAILABLE: return
        try:
            self.agent = DQNAgent(119, 53, device="cpu")
            weight_path = "agent/weights/badaamsath_dueling_v4.pth"
            if os.path.exists(weight_path):
                self.agent.load_weights(weight_path)
                self.agent.policy_net.eval()
                self.ok = True
        except Exception as e:
            print("Bot disabled:", e)

    def move(self, state, hand):
        if not self.ok: return None
        legal = state.legal_moves(hand)
        if not legal: return None
        return self.agent.select_action(state.to_tensor(self.agent.device), legal, 0.0)

# ======================
# UI
# ======================
class GameUI:
    def __init__(self, page: ft.Page):
            self.page = page
            self.state = GameState()
            self.bot = Bot()
            
            # 1. Set alignment to CENTER for the row containing the suits
            self.board_row = ft.Row(
                scroll=ft.ScrollMode.ALWAYS, 
                vertical_alignment=ft.CrossAxisAlignment.START,
                alignment=ft.MainAxisAlignment.CENTER, # Centering the suits horizontally
                spacing=15
            )
            
            self.hand_row = ft.Row(
                scroll=ft.ScrollMode.ALWAYS, 
                height=100,
                alignment=ft.MainAxisAlignment.CENTER # Centering your cards horizontally
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

        self.page.add(
            ft.Column([
                ft.Row([ft.Text("Select Hand", size=18, weight="bold"), self.count], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Divider(height=10, thickness=1),
                ft.Container(self.setup_rows, expand=True), 
                ft.ElevatedButton("Start Game", on_click=self._start, width=float("inf")), 
            ], expand=True)
        )

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
                # Status Header
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
                
                # 2. Centering the Board Area
                ft.Container(
                    # Wrapping in a Column with horizontal centering
                    content=ft.Column(
                        [self.board_row], 
                        scroll=ft.ScrollMode.AUTO,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER # Center the board row within the container
                    ),
                    expand=True,
                    padding=10,
                    alignment=ft.alignment.center, # Aligns the content of the container to the center
                    bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.WHITE),
                    border_radius=10,
                ), 
                
                ft.Divider(height=1, thickness=1),
                
                # 3. Centering the Hand Area
                ft.Container(
                    content=ft.Column([
                        ft.Text("YOUR HAND", weight="bold", size=14),
                        self.hand_row,
                    ], 
                    spacing=5,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER # Center text and cards
                    ),
                    padding=ft.padding.only(left=10, right=10, bottom=20, top=10),
                ),
            ], expand=True)

    def refresh(self):
        self.board_row.controls.clear()

        # Build each suit as a vertical column
        for suit in range(4):
            suit_stack = []
            
            # Suit Symbol at top
            suit_stack.append(
                ft.Container(
                    ft.Text(SUIT_SYMBOL[SUITS[suit]], size=20, weight="bold", color=SUIT_COLORS[SUITS[suit]]),
                    alignment=ft.alignment.center,
                    height=30
                )
            )

            low, high = self.state.table_bounds[suit]

            for r in range(13):
                cid = suit * 13 + r
                is_played = cid in self.state.played_cards
                is_playable = False
                
                if not is_played:
                    if low is None:
                        if r == SEVEN_RANK: is_playable = True
                    else:
                        if r == low - 1 or r == high + 1: is_playable = True

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
        
        suggestion = self.bot.move(self.state, self.state.my_hand)
        if suggestion is not None:
            r, s = card_parts(suggestion)
            self.info.value = f"Bot: {r}{SUIT_SYMBOL[s]}"
            self.info.color = ft.Colors.GREEN
        else:
            self.info.value = "Pass"
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
        suggestion = self.bot.move(self.state, self.state.my_hand)

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
                    border=ft.border.all(2 if is_suggested else 1, ft.Colors.GREEN if is_suggested else ft.Colors.GREY_400),
                    border_radius=6,
                    alignment=ft.alignment.center,
                    on_click=lambda _, c=cid: self._play(c, is_user=True),
                )
            )

# ======================
# ENTRY
# ======================
def main(page: ft.Page):
    page.title = "Badaam Sath Bot"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 5
    GameUI(page)

if __name__ == "__main__":
    ft.app(target=main, view=ft.WEB_BROWSER, port=8550, host="0.0.0.0")