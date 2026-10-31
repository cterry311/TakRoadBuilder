"""GUI viewer for stepping through 6x6 Tak games stored as PTN files.

Usage:
    python visualizer/ptn_viewer.py [path/to/game.ptn]

Controls: Left/Right step one ply, Home/End jump to start/end,
click a square to inspect its full stack, click a move in the list to jump to it.
"""
import os
import re
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk




import tak_engine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine_clients.tiltak_bot import Tiltak

import torch
import torch.nn.functional as F
from model.tak_net import TakNet

SIZE = 6
FLATS = 30
CAPS = 1
FILES = "abcdef"

RESULTS = {"R-0", "0-R", "F-0", "0-F", "1-0", "0-1", "1/2-1/2", "0-0"}
MOVE_RE = re.compile(r"^([1-6]?)([CS]?)([a-f])([1-6])([<>+-]?)(\d*)$")
DIRS = {"+": (0, 1), "-": (0, -1), ">": (1, 0), "<": (-1, 0)}

# colours — app chrome
APP_BG = "#16171b"
PANEL_BG = "#1d1f24"
CARD_BG = "#25282f"
CARD_HL = "#30343d"
BUTTON_BG = "#2c3038"
BUTTON_HOVER = "#3a3f4a"
TEXT = "#e9e9ec"
MUTED = "#8b909b"
ACCENT = "#e6a756"
WIN_BG = "#2f5d3a"
ERROR_FG = "#ef7b6b"

# colours — board and pieces
# FRAME = "#5b3d26"
# FRAME_EDGE = "#3d2818"
# FRAME_TEXT = "#d9c2a3"
# SQUARE_A = "#dcb98a"
# SQUARE_B = "#d2ad7b"
# GRID = "#a8825a"
# LAST_MOVE = "#f2c94c"
# ROAD_COLOR = "#6fcf73"
# SELECT_COLOR = "#58a6ff"
# WHITE_FACE, WHITE_SIDE, WHITE_EDGE = "#f6f0e3", "#d6cbb4", "#9e937d"
# BLACK_FACE, BLACK_SIDE, BLACK_EDGE = "#3a3c43", "#25272c", "#121316"
FRAME = "#2b3d56"
FRAME_EDGE = "#212131"
FRAME_TEXT = "#d9c2a3"
SQUARE_A = "#8ab9dc"
SQUARE_B = "#7badd2"
GRID = "#5882aa"
LAST_MOVE = "#f2b94c"
ROAD_COLOR = "#6fcf73"
SELECT_COLOR = "#ffa650"
WHITE_FACE, WHITE_SIDE, WHITE_EDGE = "#f6f0e3", "#d6cbb4", "#9e937d"
BLACK_FACE, BLACK_SIDE, BLACK_EDGE = "#3a3c43", "#25272c", "#121316"

EVAL_AREA_W = 170   # px reserved left of the board for the eval bars
MAX_LAYERS = 10     # stack layers drawn on a square; taller stacks are compressed to this


# ---------------------------------------------------------------- evaluation
#
# Each evaluator takes a TPS string and returns a float in [-1, 1] from WHITE's
# point of view (+1 = white winning, -1 = black winning, 0 = even), or None if
# no eval is available. If your source gives an eval from the side-to-move's
# perspective, flip the sign when black is to move (tps.split()[1] == "2").
# Evaluators run on a background thread, so they can be slow (e.g. an engine
# search) without freezing the GUI. Each one only ever runs on its own thread,
# so it's fine for it to hold state like a model or an engine subprocess.
# Evaluators are never called on positions where the game is over; the bars
# show the actual result there instead.

# info = torch.load(f'model_checkpoints/version_6_1.pt')
# _TAK_MODEL = TakNet(**info['params'])
# _TAK_MODEL.load_state_dict(info['weights'])
from engine_clients.roadbuilder import RoadBuilder
_ROADBUILDER = RoadBuilder('model_checkpoints/version_6_1.pt')

def evaluate_model(tps: str):
    # >>> INSERT MODEL EVAL HERE <<<
    # e.g. load your network once at module level, then:
    #   game = tak_engine.Tak.from_tps(tps)
    #   value = net(torch.from_numpy(game.to_tensor()).unsqueeze(0)) ...
    #   return value if tps.split()[1] == "1" else -value

    # tak_game = tak_engine.Tak.from_tps(tps)
    # _TAK_MODEL.eval()
    # with torch.no_grad():
    #     tensor_rep = torch.from_numpy(tak_game.to_tensor())
    #     tensor_rep = tensor_rep.reshape(1, 25, 6, 6)
    #     _, values = _TAK_MODEL(tensor_rep)
    #     probs = torch.nn.functional.softmax(values, dim=1).flatten()
    #     pos_value = probs[2].item() - probs[0].item()
    #     if tps.split(' ')[1] == '2':
    #         pos_value *= -1
    # return pos_value

    value = _ROADBUILDER.best_move(tps, max_sims=20_000, eps=0)[3]
    if tps.split(' ')[1] == '2':
        value *= -1
    return value

TILTAK_MOVETIME_MS = 2000   # search time per position; higher = stronger but slower eval
_tiltak = None


def evaluate_tiltak(tps):
    # >>> TILTAK EVAL <<<
    # Tiltak's value is (win - loss) / 1000 from its wdl estimate, side-to-move perspective.
    global _tiltak
    if _tiltak is None or _tiltak.p.poll() is not None:   # start (or restart) the engine lazily
        _tiltak = Tiltak()
    value = _tiltak.evaluate(position=tps, movetime_ms=TILTAK_MOVETIME_MS)["value"]
    if value is None:
        return None
    return value if tps.split()[1] == "1" else -value


# (label shown above the bar, eval function) — the bars are drawn in this order
EVALUATORS = [
    ("Model", evaluate_model),
    ("Tiltak", evaluate_tiltak),
]


class EvalWorker:
    """Runs one evaluator on a background thread, caching results by TPS.

    Only the most recently requested position is queued, so stepping quickly
    through a game doesn't build up a backlog of stale evaluations.
    """

    def __init__(self, name, fn):
        self.name = name
        self.fn = fn
        self.cache = {}         # tps -> value (float or None)
        self.errors = {}        # tps -> error message
        self.pending = None
        self.version = 0        # bumped whenever a new result lands
        self._cond = threading.Condition()
        threading.Thread(target=self._run, daemon=True, name=f"eval-{name}").start()

    def request(self, tps):
        with self._cond:
            if tps not in self.cache and tps not in self.errors:
                self.pending = tps
                self._cond.notify()

    def get(self, tps):
        """Return ('done', value) | ('error', msg) | ('pending', None)."""
        if tps in self.cache:
            return "done", self.cache[tps]
        if tps in self.errors:
            return "error", self.errors[tps]
        return "pending", None

    def _run(self):
        while True:
            with self._cond:
                while self.pending is None:
                    self._cond.wait()
                tps, self.pending = self.pending, None
            try:
                value = self.fn(tps)
                if value is not None:
                    value = max(-1.0, min(1.0, float(value)))
                self.cache[tps] = value
            except Exception as exc:
                self.errors[tps] = f"{type(exc).__name__}: {exc}"
                print(f"[{self.name} eval] {self.errors[tps]}", file=sys.stderr)
            self.version += 1


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


# ---------------------------------------------------------------- drawing helpers

def blend(c1, c2, t):
    """Mix two #rrggbb colours: t=0 gives c1, t=1 gives c2."""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def round_rect(canvas, x0, y0, x1, y1, r, **kw):
    r = max(0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
           x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
    return canvas.create_polygon(pts, smooth=True, **kw)


def piece_colours(colour):
    """(face, side, edge) colours for a player's pieces."""
    return (WHITE_FACE, WHITE_SIDE, WHITE_EDGE) if colour == 1 else (BLACK_FACE, BLACK_SIDE, BLACK_EDGE)


def ply_label(i, move):
    return f"{i // 2 + 1}{'.' if i % 2 == 0 else '...'} {move}"


class FlatButton(tk.Label):
    """A flat button with a hover state (tk.Button can't be restyled on Windows)."""

    def __init__(self, parent, text, command, font, padx=10, pady=5):
        super().__init__(parent, text=text, font=font, bg=BUTTON_BG, fg=TEXT,
                         padx=padx, pady=pady, cursor="hand2")
        self.bind("<Enter>", lambda e: self.config(bg=BUTTON_HOVER))
        self.bind("<Leave>", lambda e: self.config(bg=BUTTON_BG))
        self.bind("<Button-1>", lambda e: command())


# ---------------------------------------------------------------- GUI

class Viewer(tk.Tk):
    def __init__(self, path=None):
        super().__init__()
        self.title("Tak PTN Viewer")
        self.configure(bg=APP_BG)
        self.geometry("1240x860")
        self.minsize(940, 620)

        mono = "Cascadia Mono" if "Cascadia Mono" in tkfont.families() else "Consolas"
        self.f_title = ("Segoe UI Semibold", 14)
        self.f_body = ("Segoe UI", 10)
        self.f_small = ("Segoe UI", 9)
        self.f_header = ("Segoe UI Semibold", 8)
        self.f_status = ("Segoe UI Semibold", 12)
        self.f_icon = ("Segoe UI Symbol", 12)
        self.f_mono = (mono, 10)
        self.f_mono_small = (mono, 8)

        self.replay = None
        self.path = None
        self.ply = 0
        self.selected = None

        self.eval_workers = [EvalWorker(name, fn) for name, fn in EVALUATORS]
        self._eval_versions = None
        self._eval_ctx = None                          # (tps, result, right edge, top, bottom)
        self._bar_shown = [None] * len(self.eval_workers)  # animated bar positions
        self._anim_job = None

        self._build_widgets()
        self._bind_keys()
        if path:
            self.load(path)
        else:
            self.render()
        self.after(100, self._poll_evals)

    def _poll_evals(self):
        """Redraw the bars when a background evaluation finishes."""
        versions = [w.version for w in self.eval_workers]
        if versions != self._eval_versions:
            self._eval_versions = versions
            self._draw_evals()
        self.after(100, self._poll_evals)

    # --- layout
    def _build_widgets(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, bg=APP_BG, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda e: self.render())
        self.canvas.bind("<Button-1>", self.on_click)

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Dark.Vertical.TScrollbar", background=BUTTON_BG, troughcolor=CARD_BG,
                        bordercolor=CARD_BG, lightcolor=BUTTON_BG, darkcolor=BUTTON_BG,
                        arrowcolor=MUTED, gripcount=0)
        style.map("Dark.Vertical.TScrollbar", background=[("active", BUTTON_HOVER)])

        side = tk.Frame(self, bg=PANEL_BG, width=360)
        side.grid(row=0, column=1, sticky="ns")
        side.grid_propagate(False)
        side.columnconfigure(0, weight=1)
        side.rowconfigure(7, weight=1, minsize=180)
        pad = dict(padx=18, sticky="ew")

        self.file_label = tk.Label(side, font=self.f_title, bg=PANEL_BG, fg=TEXT, anchor="w")
        self.file_label.grid(row=0, column=0, pady=(16, 0), **pad)
        self.sub_label = tk.Label(side, font=self.f_small, bg=PANEL_BG, fg=MUTED, anchor="w")
        self.sub_label.grid(row=1, column=0, **pad)

        self._build_players(side).grid(row=2, column=0, pady=(14, 0), **pad)

        self.status_label = tk.Label(side, font=self.f_status, bg=CARD_BG, fg=TEXT,
                                     anchor="w", padx=12, pady=8)
        self.status_label.grid(row=3, column=0, pady=(10, 0), **pad)
        self.status_sub = tk.Label(side, font=self.f_small, bg=PANEL_BG, fg=MUTED,
                                   anchor="w", justify="left", wraplength=320)
        self.status_sub.grid(row=4, column=0, pady=(4, 0), **pad)

        nav = tk.Frame(side, bg=PANEL_BG)
        nav.grid(row=5, column=0, pady=(12, 0), **pad)
        nav.columnconfigure(2, weight=1)
        for col, (text, cmd) in zip((0, 1, 3, 4), [("⏮", self.first), ("◀", self.back),
                                                   ("▶", self.forward), ("⏭", self.last)]):
            FlatButton(nav, text, cmd, self.f_icon, padx=12).grid(
                row=0, column=col, padx=(0 if col == 0 else 4, 0))
        self.counter = tk.Label(nav, font=self.f_mono, bg=PANEL_BG, fg=TEXT)
        self.counter.grid(row=0, column=2, sticky="ew")

        self._section(side, "MOVES").grid(row=6, column=0, pady=(16, 4), **pad)
        moves = tk.Frame(side, bg=CARD_BG)
        moves.grid(row=7, column=0, sticky="nsew", padx=18)
        moves.rowconfigure(0, weight=1)
        moves.columnconfigure(0, weight=1)
        self.move_text = tk.Text(moves, bg=CARD_BG, fg=TEXT, font=self.f_mono, relief="flat",
                                 highlightthickness=0, padx=10, pady=8, wrap="none", cursor="arrow",
                                 spacing1=2, spacing3=2, height=8, insertwidth=0)
        self.move_text.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(moves, command=self.move_text.yview, style="Dark.Vertical.TScrollbar")
        sb.grid(row=0, column=1, sticky="ns")
        self.move_text.config(yscrollcommand=sb.set, state="disabled")
        self.move_text.tag_configure("num", foreground=MUTED)
        self.move_text.tag_configure("active_ply", background=ACCENT, foreground=PANEL_BG)

        self.stack_header = self._section(side, "STACK")
        self.stack_header.grid(row=8, column=0, pady=(16, 4), **pad)
        self.stack_canvas = tk.Canvas(side, bg=PANEL_BG, height=30, highlightthickness=0)
        self.stack_canvas.grid(row=9, column=0, **pad)

        self._section(side, "TPS").grid(row=10, column=0, pady=(14, 4), **pad)
        tps_row = tk.Frame(side, bg=PANEL_BG)
        tps_row.grid(row=11, column=0, **pad)
        tps_row.columnconfigure(0, weight=1)
        self.tps_entry = tk.Entry(tps_row, font=self.f_mono_small, readonlybackground=CARD_BG,
                                  fg=MUTED, relief="flat", highlightthickness=0)
        self.tps_entry.grid(row=0, column=0, sticky="nsew", ipady=5)
        self.tps_entry.config(state="readonly")
        FlatButton(tps_row, "Copy", self.copy_tps, self.f_small).grid(row=0, column=1, padx=(4, 0))

        files = tk.Frame(side, bg=PANEL_BG)
        files.grid(row=12, column=0, pady=(14, 16), **pad)
        for i, (text, cmd) in enumerate([("Open…", self.open_dialog),
                                         ("‹  Prev file", lambda: self.step_file(-1)),
                                         ("Next file  ›", lambda: self.step_file(1))]):
            files.columnconfigure(i, weight=1, uniform="files")
            FlatButton(files, text, cmd, self.f_body).grid(
                row=0, column=i, sticky="ew", padx=(0 if i == 0 else 4, 0))

    def _section(self, parent, text):
        return tk.Label(parent, text=text, font=self.f_header, bg=PANEL_BG, fg=MUTED, anchor="w")

    def _build_players(self, parent):
        card = tk.Frame(parent, bg=CARD_BG)
        card.columnconfigure(2, weight=1)
        for col, text in enumerate(["", "", "PLAYER", "STONES", "CAPS", "FLATS"]):
            tk.Label(card, text=text, font=self.f_header, bg=CARD_BG, fg=MUTED,
                     anchor="w" if col == 2 else "e").grid(
                row=0, column=col, sticky="ew", padx=(0, 6 if col == 5 else 0), pady=(8, 2), ipadx=4)

        self.player_rows = {}
        for colour in (1, 2):
            strip = tk.Frame(card, width=3, bg=CARD_BG)
            strip.grid(row=colour, column=0, sticky="ns")
            chip = tk.Canvas(card, width=30, height=34, bg=CARD_BG, highlightthickness=0)
            chip.grid(row=colour, column=1, sticky="w")
            face, _, edge = piece_colours(colour)
            round_rect(chip, 9, 9, 25, 25, 4, fill=face, outline=edge)
            labels = [tk.Label(card, font=self.f_body, bg=CARD_BG, fg=TEXT, anchor="w")]
            labels += [tk.Label(card, font=self.f_mono, bg=CARD_BG, fg=TEXT, anchor="e") for _ in range(3)]
            for col, lbl in enumerate(labels, start=2):
                lbl.grid(row=colour, column=col, sticky="nsew", padx=(0, 6 if col == 5 else 0), ipadx=4)
            self.player_rows[colour] = dict(strip=strip, chip=chip, labels=labels)
        tk.Frame(card, height=6, bg=CARD_BG).grid(row=3, column=0, columnspan=6)
        return card

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

    def _folder_files(self):
        folder = os.path.dirname(self.path)
        return sorted((f for f in os.listdir(folder) if f.lower().endswith(".ptn")),
                      key=lambda f: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", f)])

    def step_file(self, delta):
        if not self.path:
            return
        files = self._folder_files()
        name = os.path.basename(self.path)
        if name in files:
            idx = (files.index(name) + delta) % len(files)
            self.load(os.path.join(os.path.dirname(self.path), files[idx]))

    def load(self, path):
        try:
            replay = Replay(path)
        except Exception as exc:
            messagebox.showerror("Could not load PTN", f"{path}\n\n{exc}")
            return
        self.replay, self.path = replay, path
        self.ply = 0
        self.selected = None
        self.title(f"{os.path.basename(path)} — Tak PTN Viewer")
        self._fill_move_list()
        if replay.error:
            messagebox.showwarning("PTN problem",
                                   f"Stopped replay: {replay.error}.\n"
                                   f"Showing the {len(replay.moves)} moves before it.")
        self.render()

    def _fill_move_list(self):
        t = self.move_text
        t.config(state="normal")
        t.delete("1.0", tk.END)
        t.insert(tk.END, " start ", "ply0")
        for i, mv in enumerate(self.replay.moves):
            if i % 2 == 0:
                t.insert(tk.END, f"\n{i // 2 + 1:>3}.  ", "num")
                t.insert(tk.END, f" {mv:<9}", f"ply{i + 1}")
                t.insert(tk.END, "  ")
            else:
                t.insert(tk.END, f" {mv:<9}", f"ply{i + 1}")
        for ply in range(len(self.replay.moves) + 1):
            tag = f"ply{ply}"
            t.tag_bind(tag, "<Button-1>", lambda e, p=ply: self.goto(p))
            t.tag_bind(tag, "<Enter>", lambda e: t.config(cursor="hand2"))
            t.tag_bind(tag, "<Leave>", lambda e: t.config(cursor="arrow"))
        t.tag_raise("active_ply")
        t.config(state="disabled")

    def copy_tps(self):
        if self.replay:
            self.clipboard_clear()
            self.clipboard_append(self.replay.positions[self.ply])

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
        """Board origin and cell size; the wooden frame adds 0.4 cells on each side."""
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        avail_w = w - EVAL_AREA_W
        cell = max(12, (min(avail_w, h) - 40) / (SIZE + 0.8))
        x0 = EVAL_AREA_W + (avail_w - cell * SIZE) / 2
        y0 = (h - cell * SIZE) / 2
        return x0, y0, cell

    def render(self):
        c = self.canvas
        c.delete("all")
        if self.replay is None:
            cx, cy = c.winfo_width() / 2, c.winfo_height() / 2
            c.create_text(cx, cy - 12, fill=TEXT, font=("Segoe UI Semibold", 16),
                          text="Open a .ptn file")
            c.create_text(cx, cy + 16, fill=MUTED, font=self.f_body,
                          text="Ctrl+O  ·  ← →  step  ·  Home/End  ·  PgUp/PgDn  switch file")
            self.file_label.config(text="No file loaded")
            self._eval_ctx = None
            return

        tps = self.replay.positions[self.ply]
        board, side, move_no = parse_tps(tps)
        result = self.replay.results[self.ply]

        last_from, last_to = (None, [])
        if self.ply > 0:
            last_from, last_to = move_squares(self.replay.moves[self.ply - 1])
        road = find_road(board, result) if result in (1, 2) else set()

        x0, y0, cell = self._geometry()
        span = cell * SIZE
        pad = cell * 0.4

        # frame with a soft drop shadow
        round_rect(c, x0 - pad + 4, y0 - pad + 8, x0 + span + pad + 4, y0 + span + pad + 8,
                   pad * 0.5, fill="#0b0b0d", outline="")
        round_rect(c, x0 - pad, y0 - pad, x0 + span + pad, y0 + span + pad, pad * 0.5,
                   fill=FRAME, outline=FRAME_EDGE, width=2)
        c.create_rectangle(x0 - 3, y0 - 3, x0 + span + 3, y0 + span + 3, fill=FRAME_EDGE, outline="")

        coord_font = ("Segoe UI Semibold", max(8, int(pad * 0.36)))
        for i in range(SIZE):
            c.create_text(x0 + (i + 0.5) * cell, y0 + span + pad / 2, text=FILES[i],
                          fill=FRAME_TEXT, font=coord_font)
            c.create_text(x0 - pad / 2, y0 + (SIZE - 1 - i + 0.5) * cell, text=str(i + 1),
                          fill=FRAME_TEXT, font=coord_font)

        fills = {}
        for f in range(SIZE):
            for r in range(SIZE):
                x, y = x0 + f * cell, y0 + (SIZE - 1 - r) * cell
                fill = SQUARE_A if (f + r) % 2 else SQUARE_B
                if (f, r) in road:
                    fill = blend(fill, ROAD_COLOR, 0.6)
                elif (f, r) == last_from:
                    fill = blend(fill, LAST_MOVE, 0.6)
                elif (f, r) in last_to:
                    fill = blend(fill, LAST_MOVE, 0.6)
                fills[f, r] = fill
                c.create_rectangle(x, y, x + cell, y + cell, fill=fill, outline=GRID)
                if (f, r) == self.selected:
                    round_rect(c, x + 3, y + 3, x + cell - 3, y + cell - 3, cell * 0.08,
                               fill="", outline=SELECT_COLOR, width=3)

        # pieces drawn top rank first so taller stacks overlap the square behind them
        for r in reversed(range(SIZE)):
            for f in range(SIZE):
                x, y = x0 + f * cell, y0 + (SIZE - 1 - r) * cell
                self._draw_stack(board[f][r], x, y, cell, fills[f, r])

        self._eval_ctx = (tps, result, x0 - pad - 18, y0, y0 + span)
        self._draw_evals()
        self._update_panel(board, side, move_no, result, tps)

    def _draw_evals(self):
        """Vertical eval bars left of the board: white fills from the bottom."""
        c = self.canvas
        c.delete("evals")
        if self._eval_ctx is None:
            return
        tps, result, right, top, bottom = self._eval_ctx
        bar_w, gap = 20, 48
        n = len(self.eval_workers)
        left = right - n * bar_w - (n - 1) * gap
        animating = False
        for i, worker in enumerate(self.eval_workers):
            if result:
                # game over: show the real result, engines can't evaluate these (Tiltak crashes)
                status, value = "done", {1: 1.0, 2: -1.0, 3: 0.0}[result]
            else:
                worker.request(tps)
                status, value = worker.get(tps)

            # ease the bar towards its new value; keep the old one while a new eval is pending
            shown = self._bar_shown[i]
            if status == "done" and value is not None:
                shown = 0.0 if shown is None else shown
                if abs(value - shown) > 0.003:
                    shown += (value - shown) * 0.22
                    animating = True
                else:
                    shown = value
            elif status != "pending":
                shown = None
            self._bar_shown[i] = shown

            bx0 = left + i * (bar_w + gap)
            bx1 = bx0 + bar_w
            mx = (bx0 + bx1) / 2
            c.create_text(mx, top - 16, text=worker.name.upper(), fill=MUTED,
                          font=self.f_header, tags="evals")
            if shown is None:
                c.create_rectangle(bx0, top, bx1, bottom, fill=CARD_BG, outline="", tags="evals")
            else:
                split = bottom - (shown + 1) / 2 * (bottom - top)
                c.create_rectangle(bx0, top, bx1, split, fill=BLACK_FACE, outline="", tags="evals")
                c.create_rectangle(bx0, split, bx1, bottom, fill=WHITE_FACE, outline="", tags="evals")
            c.create_rectangle(bx0, top, bx1, bottom, outline="#0b0b0d", tags="evals")
            mid = (top + bottom) / 2
            for x_a, x_b in ((bx0 - 5, bx0 - 1), (bx1 + 1, bx1 + 5)):
                c.create_line(x_a, mid, x_b, mid, fill=MUTED, tags="evals")

            if status == "done" and value is not None:
                label = f"{value:+.2f}"
            else:
                label = {"pending": "…", "error": "err"}.get(status, "n/a")
            round_rect(c, mx - 22, bottom + 8, mx + 22, bottom + 28, 6, fill=CARD_BG,
                       outline="", tags="evals")
            c.create_text(mx, bottom + 18, text=label, font=self.f_mono_small, tags="evals",
                          fill=ERROR_FG if status == "error" else TEXT)

        if animating and self._anim_job is None:
            self._anim_job = self.after(16, self._anim_tick)

    def _anim_tick(self):
        self._anim_job = None
        self._draw_evals()

    def _draw_stack(self, stack, x, y, cell, square_fill):
        if not stack:
            return
        c = self.canvas
        shadow = blend(square_fill, "#000000", 0.32)
        face = cell * 0.46          # width of a flat's top face
        depth = cell * 0.07         # visible thickness of a flat
        sh = cell * 0.035           # shadow offset
        layers = stack[-MAX_LAYERS:]
        n = len(layers)
        step = min(cell * 0.06, cell * 0.3 / (n - 1)) if n > 1 else 0
        cx = x + cell / 2
        base = y + cell / 2 + (face + depth + (n - 1) * step) / 2   # bottom edge of the lowest piece
        radius = face * 0.14

        def plate(level, colour, cast_shadow):
            top = base - depth - face - level * step
            fc, sc, ec = piece_colours(colour)
            x_a, x_b = cx - face / 2, cx + face / 2
            if cast_shadow:
                round_rect(c, x_a + sh, top + depth + sh, x_b + sh, top + face + depth + sh,
                           radius, fill=shadow, outline="")
            round_rect(c, x_a, top + depth, x_b, top + face + depth, radius, fill=sc, outline=ec)
            round_rect(c, x_a, top, x_b, top + face, radius, fill=fc, outline=ec)

        for level, (colour, _) in enumerate(layers[:-1]):
            plate(level, colour, level == 0)

        colour, kind = layers[-1]
        fc, sc, ec = piece_colours(colour)
        level = n - 1
        cy = base - depth - face / 2 - level * step
        if kind == "F":
            plate(level, colour, n == 1)
        elif kind == "S":
            # wall: a slab lying diagonally across the square, extruded downwards
            a, b = face * 0.6, face * 0.17
            ux, uy, vx, vy = 0.7071 * a, -0.7071 * a, 0.7071 * b, 0.7071 * b
            pts = [(cx - ux - vx, cy - uy - vy), (cx + ux - vx, cy + uy - vy),
                   (cx + ux + vx, cy + uy + vy), (cx - ux + vx, cy - uy + vy)]
            if n == 1:
                c.create_polygon([(px + sh, py + depth + sh) for px, py in pts], fill=shadow)
            c.create_polygon([(px, py + depth) for px, py in pts], fill=sc, outline=ec)
            c.create_polygon(pts, fill=fc, outline=ec)
        else:
            # capstone: a short cylinder seen from above
            rr, h = face * 0.48, depth * 1.4
            if n == 1:
                c.create_oval(cx - rr + sh, cy - rr + h + sh, cx + rr + sh, cy + rr + h + sh,
                              fill=shadow, outline="")
            c.create_oval(cx - rr, cy - rr + h, cx + rr, cy + rr + h, fill=sc, outline=ec)
            c.create_rectangle(cx - rr, cy, cx + rr, cy + h, fill=sc, outline="")
            c.create_line(cx - rr, cy, cx - rr, cy + h, fill=ec)
            c.create_line(cx + rr, cy, cx + rr, cy + h, fill=ec)
            c.create_oval(cx - rr, cy - rr, cx + rr, cy + rr, fill=fc, outline=ec)
            c.create_oval(cx - rr * 0.55, cy - rr * 0.55, cx + rr * 0.55, cy + rr * 0.55,
                          outline=blend(fc, ec, 0.6), width=max(1, cell * 0.015))

        if len(stack) > 1:
            text = str(len(stack))
            bw, bh = cell * (0.13 + 0.07 * len(text)), cell * 0.17
            bx1, by0 = x + cell - cell * 0.05, y + cell * 0.05
            round_rect(c, bx1 - bw, by0, bx1, by0 + bh, bh / 2, fill=PANEL_BG, outline="")
            c.create_text(bx1 - bw / 2, by0 + bh / 2, text=text, fill=TEXT,
                          font=("Segoe UI Semibold", max(7, int(cell * 0.085))))

    def _update_panel(self, board, side, move_no, result, tps):
        h = self.replay.headers
        names = {1: h.get("player1", "White"), 2: h.get("player2", "Black")}
        n = len(self.replay.moves)

        self.file_label.config(text=os.path.basename(self.path))
        files = self._folder_files()
        name = os.path.basename(self.path)
        position = f"  ·  file {files.index(name) + 1} of {len(files)}" if name in files else ""
        self.sub_label.config(text=f"{os.path.basename(os.path.dirname(os.path.abspath(self.path)))}"
                                   f"{position}  ·  {n} plies")

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

        for colour, row in self.player_rows.items():
            if result == 0:
                active, strip = colour == side, ACCENT
            else:
                active, strip = colour == result, ROAD_COLOR
            bg = CARD_HL if active else CARD_BG
            row["strip"].config(bg=strip if active else CARD_BG)
            row["chip"].config(bg=bg)
            values = [names[colour], FLATS - used[colour][0], CAPS - used[colour][1], top_flats[colour]]
            for lbl, val in zip(row["labels"], values):
                lbl.config(text=str(val), bg=bg)

        if result == 0:
            status, bg = f"Move {move_no}  ·  {'White' if side == 1 else 'Black'} to move", CARD_BG
        elif result == 3:
            status, bg = "Draw", CARD_HL
        else:
            how = "road" if find_road(board, result) else "flats"
            status, bg = f"{'White' if result == 1 else 'Black'} wins by {how}", WIN_BG
        self.status_label.config(text=status, bg=bg)

        last = "—" if self.ply == 0 else ply_label(self.ply - 1, self.replay.moves[self.ply - 1])
        sub = f"Last move  {last}   ·   {empty} empty"
        if self.replay.error:
            sub += f"\n⚠ {self.replay.error}"
        self.status_sub.config(text=sub, fg=ERROR_FG if self.replay.error else MUTED)
        self.counter.config(text=f"{self.ply} / {n}")

        t = self.move_text
        t.tag_remove("active_ply", "1.0", tk.END)
        ranges = t.tag_ranges(f"ply{self.ply}")
        if ranges:
            t.tag_add("active_ply", *ranges)
            t.see(ranges[0])

        self._draw_stack_inspector(board)

        self.tps_entry.config(state="normal")
        self.tps_entry.delete(0, tk.END)
        self.tps_entry.insert(0, tps)
        self.tps_entry.config(state="readonly")

    def _draw_stack_inspector(self, board):
        c = self.stack_canvas
        c.delete("all")
        width = max(c.winfo_width(), 300)
        if not self.selected:
            self.stack_header.config(text="STACK")
            c.config(height=24)
            c.create_text(0, 12, anchor="w", text="Click a square to inspect its stack",
                          fill=MUTED, font=self.f_small)
            return
        f, r = self.selected
        stack = board[f][r]
        self.stack_header.config(text=f"STACK  ·  {FILES[f]}{r + 1}  ·  {len(stack)} high"
                                      + ("  ·  bottom → top" if stack else ""))
        if not stack:
            c.config(height=24)
            c.create_text(0, 12, anchor="w", text="Empty square", fill=MUTED, font=self.f_small)
            return
        chip_w, chip_h, gap = 14, 22, 4
        per_row = max(1, int((width + gap) // (chip_w + gap)))
        rows = (len(stack) + per_row - 1) // per_row
        c.config(height=rows * (chip_h + gap) + 2)
        for i, (colour, kind) in enumerate(stack):
            fc, _, ec = piece_colours(colour)
            x = (i % per_row) * (chip_w + gap)
            y = (i // per_row) * (chip_h + gap) + 1
            if kind == "C":
                c.create_oval(x, y + 4, x + chip_w, y + 4 + chip_w, fill=fc, outline=ec)
            elif kind == "S":
                c.create_rectangle(x + 4, y, x + chip_w - 4, y + chip_h, fill=fc, outline=ec)
            else:
                round_rect(c, x, y, x + chip_w, y + chip_h, 3, fill=fc, outline=ec)


if __name__ == "__main__":
    Viewer(sys.argv[1] if len(sys.argv) > 1 else None).mainloop()
