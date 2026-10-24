"""GUI viewer for stepping through 6x6 Tak games stored as PTN files.

Usage:
    python visualizer/ptn_viewer.py [path/to/game.ptn]

Controls: Left/Right step one ply, Home/End jump to start/end,
click a square to inspect its full stack, click a move in the list to jump to it.
"""
import os
import re
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

import tak_engine

SIZE = 6
FLATS = 30
CAPS = 1
FILES = "abcdef"

RESULTS = {"R-0", "0-R", "F-0", "0-F", "1-0", "0-1", "1/2-1/2", "0-0"}
MOVE_RE = re.compile(r"^([1-6]?)([CS]?)([a-f])([1-6])([<>+-]?)(\d*)$")
DIRS = {"+": (0, 1), "-": (0, -1), ">": (1, 0), "<": (-1, 0)}

# colours
BOARD_BG = "#3b2f25"
SQUARE_A = "#c9a675"
SQUARE_B = "#bf9a68"
HIGHLIGHT_FROM = "#e3c35a"
HIGHLIGHT_TO = "#f0d77a"
ROAD_COLOR = "#7fb36a"
SELECT_COLOR = "#5aa0e3"
WHITE_FILL, WHITE_EDGE = "#f4eee0", "#8c826f"
BLACK_FILL, BLACK_EDGE = "#2b2b2b", "#000000"
PANEL_BG = "#262626"
PANEL_FG = "#e8e8e8"
MUTED_FG = "#a0a0a0"


# ---------------------------------------------------------------- parsing

def parse_ptn(text):
    """Return (headers dict, list of raw move tokens)."""
    headers = {}
    for key, value in re.findall(r'\[\s*(\w+)\s+"([^"]*)"\s*\]', text):
        headers[key.lower()] = value
    body = re.sub(r"\[[^\]]*\]", " ", text)
    body = re.sub(r"\{[^}]*\}", " ", body)
    moves = []
    for tok in body.split():
        if re.fullmatch(r"\d+\.+", tok) or tok in RESULTS:
            continue
        tok = re.sub(r"^\d+\.+", "", tok).rstrip("'\"!?*")
        if tok:
            moves.append(tok)
    return headers, moves


def parse_tps(tps):
    """Return (board, side_to_move, move_number).

    board[file][rank] is a list of (colour, kind) from bottom to top, where
    colour is 1/2 and kind is 'F', 'S' or 'C'.
    """
    rows, side, move_no = tps.split()
    board = [[[] for _ in range(SIZE)] for _ in range(SIZE)]
    for i, row in enumerate(rows.split("/")):
        rank = SIZE - 1 - i
        f = 0
        for item in row.split(","):
            if item.startswith("x"):
                f += int(item[1:] or 1)
                continue
            stack = [(int(c), "F") for c in item if c in "12"]
            if item[-1] in "SC":
                stack[-1] = (stack[-1][0], item[-1])
            board[f][rank] = stack
            f += 1
    return board, int(side), int(move_no)


def move_squares(ptn):
    """Squares touched by a move: (origin, [destination squares])."""
    m = MOVE_RE.match(ptn)
    if not m:
        return None, []
    count, _, file, rank, direction, drops = m.groups()
    sq = (FILES.index(file), int(rank) - 1)
    if not direction:
        return sq, []
    n_steps = len(drops) if drops else 1
    dx, dy = DIRS[direction]
    return sq, [(sq[0] + dx * k, sq[1] + dy * k) for k in range(1, n_steps + 1)]


def find_road(board, colour):
    """Return the set of squares forming a road for `colour`, or empty set."""
    def is_road(f, r):
        st = board[f][r]
        return bool(st) and st[-1][0] == colour and st[-1][1] != "S"

    for axis in (0, 1):
        # axis 0: connect file a to file f; axis 1: rank 1 to rank 6
        starts = [(0, i) if axis == 0 else (i, 0) for i in range(SIZE)]
        seen, frontier, parent = set(), [], {}
        for s in starts:
            if is_road(*s):
                seen.add(s)
                frontier.append(s)
                parent[s] = None
        while frontier:
            cur = frontier.pop(0)
            if cur[axis] == SIZE - 1:
                path, node = set(), cur
                while node is not None:
                    path.add(node)
                    node = parent[node]
                return path
            for dx, dy in DIRS.values():
                nxt = (cur[0] + dx, cur[1] + dy)
                if 0 <= nxt[0] < SIZE and 0 <= nxt[1] < SIZE and nxt not in seen and is_road(*nxt):
                    seen.add(nxt)
                    parent[nxt] = cur
                    frontier.append(nxt)
    return set()


class Replay:
    """A PTN game replayed through tak_engine, one TPS snapshot per ply."""

    def __init__(self, path):
        with open(path, encoding="utf-8") as fh:
            self.headers, tokens = parse_ptn(fh.read())
        size = self.headers.get("size", "6")
        if size != str(SIZE):
            raise ValueError(f"only size 6 games are supported (file says {size})")

        game = tak_engine.Tak()
        self.moves = []          # normalised ptn per ply
        self.positions = [game.to_tps()]
        self.results = [game.result()]
        self.error = None
        for i, tok in enumerate(tokens):
            ply_label = f"{i // 2 + 1}{'.' if i % 2 == 0 else '...'} {tok}"
            if game.result() != 0:
                self.error = f"moves continue after the game ended at {ply_label}"
                break
            if not MOVE_RE.match(tok):
                self.error = f"unparseable move {ply_label}"
                break
            raw = tak_engine.Tak.ptn_to_raw(tok)
            if raw not in game.legal_moves_raw():
                self.error = f"illegal move {ply_label}"
                break
            game.make_move(tok)
            self.moves.append(tak_engine.Tak.raw_to_ptn(raw))
            self.positions.append(game.to_tps())
            self.results.append(game.result())


# ---------------------------------------------------------------- GUI

class Viewer(tk.Tk):
    def __init__(self, path=None):
        super().__init__()
        self.title("Tak PTN Viewer")
        self.configure(bg=PANEL_BG)
        self.geometry("1100x720")
        self.minsize(800, 520)

        self.replay = None
        self.path = None
        self.ply = 0
        self.selected = None

        self._build_widgets()
        self._bind_keys()
        if path:
            self.load(path)
        else:
            self.render()

    # --- layout
    def _build_widgets(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, bg=BOARD_BG, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda e: self.render())
        self.canvas.bind("<Button-1>", self.on_click)

        side = tk.Frame(self, bg=PANEL_BG, width=330)
        side.grid(row=0, column=1, sticky="ns", padx=(0, 10), pady=10)
        side.grid_propagate(False)
        side.columnconfigure(0, weight=1)
        side.rowconfigure(3, weight=1)

        lbl = dict(bg=PANEL_BG, fg=PANEL_FG, anchor="w", justify="left")
        self.file_label = tk.Label(side, font=("Segoe UI", 11, "bold"), wraplength=310, **lbl)
        self.file_label.grid(row=0, column=0, sticky="ew")

        self.info = tk.Label(side, font=("Consolas", 11), **lbl)
        self.info.grid(row=1, column=0, sticky="ew", pady=(6, 6))

        nav = tk.Frame(side, bg=PANEL_BG)
        nav.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        for i, (text, cmd) in enumerate([("⏮", self.first), ("◀", self.back),
                                         ("▶", self.forward), ("⏭", self.last)]):
            nav.columnconfigure(i, weight=1)
            tk.Button(nav, text=text, command=cmd, font=("Segoe UI", 14),
                      width=3).grid(row=0, column=i, sticky="ew", padx=2)

        list_frame = tk.Frame(side, bg=PANEL_BG)
        list_frame.grid(row=3, column=0, sticky="nsew")
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        self.move_list = tk.Listbox(list_frame, font=("Consolas", 11), bg="#1c1c1c", fg=PANEL_FG,
                                    selectbackground=SELECT_COLOR, activestyle="none",
                                    exportselection=False, highlightthickness=0)
        self.move_list.grid(row=0, column=0, sticky="nsew")
        sb = tk.Scrollbar(list_frame, command=self.move_list.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.move_list.config(yscrollcommand=sb.set)
        self.move_list.bind("<<ListboxSelect>>", self.on_list_select)

        self.stack_label = tk.Label(side, font=("Consolas", 10), fg=MUTED_FG, bg=PANEL_BG,
                                    anchor="w", justify="left", wraplength=310)
        self.stack_label.grid(row=4, column=0, sticky="ew", pady=(8, 0))

        self.tps_entry = tk.Entry(side, font=("Consolas", 9), readonlybackground="#1c1c1c",
                                  fg=MUTED_FG, relief="flat")
        self.tps_entry.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        self.tps_entry.config(state="readonly")

        files = tk.Frame(side, bg=PANEL_BG)
        files.grid(row=6, column=0, sticky="ew", pady=(8, 0))
        for i, (text, cmd) in enumerate([("Open…", self.open_dialog),
                                         ("Prev file", lambda: self.step_file(-1)),
                                         ("Next file", lambda: self.step_file(1))]):
            files.columnconfigure(i, weight=1)
            tk.Button(files, text=text, command=cmd).grid(row=0, column=i, sticky="ew", padx=2)

    def _bind_keys(self):
        self.bind("<Left>", lambda e: self.back())
        self.bind("<Right>", lambda e: self.forward())
        self.bind("<Home>", lambda e: self.first())
        self.bind("<End>", lambda e: self.last())
        self.bind("<Prior>", lambda e: self.step_file(-1))
        self.bind("<Next>", lambda e: self.step_file(1))
        self.bind("<Control-o>", lambda e: self.open_dialog())

    # --- file handling
    def open_dialog(self):
        start = os.path.dirname(self.path) if self.path else os.path.join(os.getcwd(), "ptn_files")
        path = filedialog.askopenfilename(initialdir=start if os.path.isdir(start) else os.getcwd(),
                                          filetypes=[("PTN files", "*.ptn"), ("All files", "*.*")])
        if path:
            self.load(path)

    def step_file(self, delta):
        if not self.path:
            return
        folder = os.path.dirname(self.path)
        files = sorted((f for f in os.listdir(folder) if f.lower().endswith(".ptn")),
                       key=lambda f: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", f)])
        name = os.path.basename(self.path)
        if name in files:
            idx = (files.index(name) + delta) % len(files)
            self.load(os.path.join(folder, files[idx]))

    def load(self, path):
        try:
            replay = Replay(path)
        except Exception as exc:
            messagebox.showerror("Could not load PTN", f"{path}\n\n{exc}")
            return
        self.replay, self.path = replay, path
        self.ply = 0
        self.selected = None
        self.move_list.delete(0, tk.END)
        self.move_list.insert(tk.END, "   start")
        for i, mv in enumerate(replay.moves):
            prefix = f"{i // 2 + 1:>3}." if i % 2 == 0 else "    …"
            self.move_list.insert(tk.END, f"{prefix} {mv}")
        if replay.error:
            messagebox.showwarning("PTN problem",
                                   f"Stopped replay: {replay.error}.\n"
                                   f"Showing the {len(replay.moves)} moves before it.")
        self.render()

    # --- navigation
    def goto(self, ply):
        if self.replay is None:
            return
        self.ply = max(0, min(ply, len(self.replay.moves)))
        self.render()

    def first(self):
        self.goto(0)

    def last(self):
        self.goto(len(self.replay.moves) if self.replay else 0)

    def forward(self):
        self.goto(self.ply + 1)

    def back(self):
        self.goto(self.ply - 1)

    def on_list_select(self, _event):
        sel = self.move_list.curselection()
        if sel and sel[0] != self.ply:
            self.goto(sel[0])

    def on_click(self, event):
        if self.replay is None:
            return
        x0, y0, cell = self._geometry()
        f = int((event.x - x0) // cell)
        r = SIZE - 1 - int((event.y - y0) // cell)
        if 0 <= f < SIZE and 0 <= r < SIZE:
            self.selected = None if self.selected == (f, r) else (f, r)
            self.render()

    # --- drawing
    def _geometry(self):
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        margin = 34
        cell = max(10, (min(w, h) - 2 * margin) / SIZE)
        x0 = (w - cell * SIZE) / 2
        y0 = (h - cell * SIZE) / 2
        return x0, y0, cell

    def render(self):
        c = self.canvas
        c.delete("all")
        if self.replay is None:
            c.create_text(c.winfo_width() / 2, c.winfo_height() / 2, fill=PANEL_FG,
                          font=("Segoe UI", 14), text="Open a .ptn file (Ctrl+O)")
            self.file_label.config(text="No file loaded")
            return

        tps = self.replay.positions[self.ply]
        board, side, move_no = parse_tps(tps)
        result = self.replay.results[self.ply]

        last_from, last_to = (None, [])
        if self.ply > 0:
            last_from, last_to = move_squares(self.replay.moves[self.ply - 1])
        road = set()
        if result in (1, 2):
            road = find_road(board, result)

        x0, y0, cell = self._geometry()
        font_px = max(8, int(cell * 0.14))
        for f in range(SIZE):
            for r in range(SIZE):
                x = x0 + f * cell
                y = y0 + (SIZE - 1 - r) * cell
                fill = SQUARE_A if (f + r) % 2 else SQUARE_B
                if (f, r) in road:
                    fill = ROAD_COLOR
                elif (f, r) == last_from:
                    fill = HIGHLIGHT_FROM
                elif (f, r) in last_to:
                    fill = HIGHLIGHT_TO
                c.create_rectangle(x, y, x + cell, y + cell, fill=fill, outline=BOARD_BG, width=2)
                if (f, r) == self.selected:
                    c.create_rectangle(x + 3, y + 3, x + cell - 3, y + cell - 3,
                                       outline=SELECT_COLOR, width=3)
                self._draw_stack(board[f][r], x, y, cell, font_px)

        for i in range(SIZE):
            c.create_text(x0 + (i + 0.5) * cell, y0 + SIZE * cell + 16, text=FILES[i],
                          fill=PANEL_FG, font=("Segoe UI", 12))
            c.create_text(x0 - 16, y0 + (SIZE - 1 - i + 0.5) * cell, text=str(i + 1),
                          fill=PANEL_FG, font=("Segoe UI", 12))

        self._update_panel(board, side, move_no, result, tps)

    def _draw_stack(self, stack, x, y, cell, font_px):
        if not stack:
            return
        c = self.canvas
        # side column showing the stack from bottom (bottom of square) to top
        shown = stack[-12:]
        bar_h = (cell * 0.8) / 12
        bx0, bx1 = x + cell * 0.06, x + cell * 0.18
        for i, (colour, _) in enumerate(shown):
            by1 = y + cell * 0.9 - i * bar_h
            fill, edge = (WHITE_FILL, WHITE_EDGE) if colour == 1 else (BLACK_FILL, WHITE_EDGE)
            c.create_rectangle(bx0, by1 - bar_h + 1, bx1, by1, fill=fill, outline=edge)

        colour, kind = stack[-1]
        fill, edge = (WHITE_FILL, WHITE_EDGE) if colour == 1 else (BLACK_FILL, BLACK_EDGE)
        cx, cy = x + cell * 0.56, y + cell * 0.52
        if kind == "F":
            s = cell * 0.22
            c.create_rectangle(cx - s, cy - s, cx + s, cy + s, fill=fill, outline=edge, width=2)
        elif kind == "S":
            # wall drawn as a slanted slab
            a, b = cell * 0.25, cell * 0.07
            c.create_polygon(cx - a, cy + a - b, cx - a + b, cy + a, cx + a, cy - a + b,
                             cx + a - b, cy - a, fill=fill, outline=edge, width=2)
        else:
            s = cell * 0.21
            c.create_oval(cx - s, cy - s, cx + s, cy + s, fill=fill, outline=edge, width=2)
            c.create_oval(cx - s * 0.45, cy - s * 0.45, cx + s * 0.45, cy + s * 0.45,
                          outline=edge, width=2)

        if len(stack) > 1:
            c.create_text(x + cell * 0.95, y + cell * 0.03, text=str(len(stack)), anchor="ne",
                          fill="#1a1a1a", font=("Segoe UI", font_px, "bold"))

    def _update_panel(self, board, side, move_no, result, tps):
        h = self.replay.headers
        p1, p2 = h.get("player1", "White"), h.get("player2", "Black")
        self.file_label.config(text=f"{os.path.basename(self.path)}\n{p1} (W) vs {p2} (B)")

        used = {1: [0, 0], 2: [0, 0]}  # [stones, caps]
        top_flats = {1: 0, 2: 0}
        empty = 0
        for f in range(SIZE):
            for r in range(SIZE):
                st = board[f][r]
                if not st:
                    empty += 1
                    continue
                for colour, kind in st:
                    used[colour][1 if kind == "C" else 0] += 1
                if st[-1][1] == "F":
                    top_flats[st[-1][0]] += 1

        n = len(self.replay.moves)
        if self.ply == 0:
            last = "—"
        else:
            i = self.ply - 1
            last = f"{i // 2 + 1}{'.' if i % 2 == 0 else '...'} {self.replay.moves[i]}"
        status = {0: f"{'White' if side == 1 else 'Black'} to move",
                  1: "White wins", 2: "Black wins", 3: "Draw"}[result]
        if result in (1, 2):
            status += " (road)" if find_road(board, result) else " (flats)"

        lines = [
            f"Ply      {self.ply} / {n}",
            f"Last     {last}",
            f"Turn     {move_no}",
            f"Status   {status}",
            "",
            f"           White   Black",
            f"Stones     {FLATS - used[1][0]:>5}   {FLATS - used[2][0]:>5}",
            f"Caps       {CAPS - used[1][1]:>5}   {CAPS - used[2][1]:>5}",
            f"Flats      {top_flats[1]:>5}   {top_flats[2]:>5}",
            f"Empty sq   {empty:>5}",
        ]
        if self.replay.error:
            lines += ["", f"! {self.replay.error}"]
        self.info.config(text="\n".join(lines))

        self.move_list.selection_clear(0, tk.END)
        self.move_list.selection_set(self.ply)
        self.move_list.see(self.ply)

        if self.selected:
            f, r = self.selected
            st = board[f][r]
            desc = " ".join(("W" if col == 1 else "B") + ("" if k == "F" else k) for col, k in st)
            self.stack_label.config(text=f"{FILES[f]}{r + 1} (bottom→top): {desc or 'empty'}")
        else:
            self.stack_label.config(text="Click a square to inspect its stack.")

        self.tps_entry.config(state="normal")
        self.tps_entry.delete(0, tk.END)
        self.tps_entry.insert(0, tps)
        self.tps_entry.config(state="readonly")


if __name__ == "__main__":
    Viewer(sys.argv[1] if len(sys.argv) > 1 else None).mainloop()
