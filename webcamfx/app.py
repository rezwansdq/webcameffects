"""WebcamFX window: live video on the left, control panel on the right."""

import tkinter as tk
from tkinter import colorchooser, ttk

import cv2
from PIL import Image, ImageTk

from .engine import QUALITY_WIDTHS, Engine, Settings

BG = "#0b0d0b"
PANEL = "#141814"
FG = "#d8e8d8"
ACCENT = "#39ff14"

PRESETS = [
    ("Toxic Green", (20, 255, 57)),
    ("Blood Red", (40, 20, 255)),
    ("Ice Blue", (255, 200, 40)),
    ("Purple", (255, 40, 180)),
    ("White", (255, 255, 255)),
]


def bgr_to_hex(bgr):
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


class App:
    def __init__(self, root):
        self.root = root
        self.s = Settings()
        self.engine = Engine(self.s)
        self.fullscreen = False
        self._photo = None

        root.title("WebcamFX")
        root.configure(bg=BG)
        root.geometry("1280x720")
        root.minsize(800, 480)

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(".", background=PANEL, foreground=FG, fieldbackground="#202620")
        style.configure("TLabel", background=PANEL, foreground=FG)
        style.configure("Head.TLabel", font=("Segoe UI", 10, "bold"), foreground=ACCENT)
        style.configure("TCheckbutton", background=PANEL, foreground=FG)
        style.map("TCheckbutton", background=[("active", PANEL)])
        style.configure("Horizontal.TScale", background=PANEL, troughcolor="#202620")
        style.map("TCombobox", fieldbackground=[("readonly", "#202620")], foreground=[("readonly", FG)],
                  selectbackground=[("readonly", "#202620")], selectforeground=[("readonly", FG)])
        root.option_add("*TCombobox*Listbox.background", "#202620")
        root.option_add("*TCombobox*Listbox.foreground", FG)

        # Panel is packed first so the video can never squeeze it.
        self.panel = tk.Frame(root, bg=PANEL, width=260, padx=14, pady=12)
        self.panel.pack(side="right", fill="y")
        self.panel.pack_propagate(False)
        self._build_panel()

        self.video = tk.Label(root, bg="black", width=1, height=1)
        self.video.pack(side="left", fill="both", expand=True)

        root.bind("<space>", lambda e: self.toggle_effects())
        root.bind("<F11>", lambda e: self.toggle_fullscreen())
        root.bind("<Escape>", lambda e: self.toggle_fullscreen(False))
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.engine.start()
        self._tick()

    # ---- panel -------------------------------------------------------------

    def _build_panel(self):
        p = self.panel
        self.toggle_btn = tk.Button(
            p, font=("Segoe UI", 16, "bold"), relief="flat", bd=0, height=2,
            cursor="hand2", command=self.toggle_effects,
        )
        self.toggle_btn.pack(fill="x", pady=(0, 12))
        self._refresh_toggle()

        ttk.Label(p, text="CAMERA", style="Head.TLabel").pack(anchor="w")
        row = tk.Frame(p, bg=PANEL)
        row.pack(fill="x", pady=(2, 10))
        self.cam_var = tk.StringVar(value=str(self.s.camera))
        cam = ttk.Combobox(row, textvariable=self.cam_var, values=[str(i) for i in range(5)],
                           width=4, state="readonly")
        cam.pack(side="left")
        cam.bind("<<ComboboxSelected>>", lambda e: setattr(self.s, "camera", int(self.cam_var.get())))
        ttk.Label(row, text="  Quality").pack(side="left")
        self.q_var = tk.StringVar(value=self.s.quality)
        q = ttk.Combobox(row, textvariable=self.q_var, values=list(QUALITY_WIDTHS), width=9, state="readonly")
        q.pack(side="left", padx=(4, 0))
        q.bind("<<ComboboxSelected>>", lambda e: setattr(self.s, "quality", self.q_var.get()))

        ttk.Label(p, text="EFFECTS", style="Head.TLabel").pack(anchor="w")
        for label, attr in [("Glow around hands", "aura_hands"), ("Glow around arms", "aura_arms"),
                            ("Glow around head", "aura_head"), ("Motion trails", "trails"),
                            ("Skeleton lines on body", "skeleton"), ("Mirror view", "mirror")]:
            self._check(label, attr)

        ttk.Label(p, text="COLOR", style="Head.TLabel").pack(anchor="w", pady=(10, 2))
        swatches = tk.Frame(p, bg=PANEL)
        swatches.pack(fill="x")
        for name, bgr in PRESETS:
            sw = tk.Frame(swatches, bg=bgr_to_hex(bgr), width=30, height=26, cursor="hand2")
            sw.pack(side="left", padx=(0, 6))
            sw.bind("<Button-1>", lambda e, c=bgr: self.set_color(c))
        custom = tk.Label(swatches, text="+", bg="#202620", fg=FG, width=3, cursor="hand2")
        custom.pack(side="left", fill="y")
        custom.bind("<Button-1>", lambda e: self.pick_color())
        self.swatch = tk.Frame(p, height=6, bg=bgr_to_hex(self.s.color_bgr))
        self.swatch.pack(fill="x", pady=(6, 10))

        ttk.Label(p, text="TUNING", style="Head.TLabel").pack(anchor="w")
        self._slider("Glow", "intensity", 0.0, 1.0)
        self._slider("Trail length", "trail_length", 0.0, 1.0)
        self._slider("Glow width", "thickness", 2.0, 16.0)
        self._slider("Background darkness", "darkness", 0.0, 1.0)

        tk.Button(p, text="Fullscreen (F11)", bg="#202620", fg=FG, relief="flat", bd=0, cursor="hand2",
                  command=self.toggle_fullscreen).pack(fill="x", pady=(12, 0), ipady=6)

        self.status = ttk.Label(p, text="", wraplength=230, justify="left")
        self.status.pack(side="bottom", anchor="w")
        ttk.Label(p, text="Space = on/off   F11 = fullscreen", foreground="#7a8a7a").pack(side="bottom", anchor="w")

    def _check(self, label, attr):
        var = tk.BooleanVar(value=getattr(self.s, attr))
        ttk.Checkbutton(self.panel, text=label, variable=var,
                        command=lambda: setattr(self.s, attr, var.get())).pack(anchor="w")

    def _slider(self, label, attr, lo, hi):
        ttk.Label(self.panel, text=label).pack(anchor="w")
        var = tk.DoubleVar(value=getattr(self.s, attr))
        ttk.Scale(self.panel, from_=lo, to=hi, variable=var,
                  command=lambda v: setattr(self.s, attr, float(v))).pack(fill="x", pady=(0, 6))

    def _refresh_toggle(self):
        on = self.s.effects_on
        self.toggle_btn.configure(
            text="EFFECTS: ON" if on else "EFFECTS: OFF",
            bg=ACCENT if on else "#2a2f2a", fg="black" if on else FG,
            activebackground=ACCENT if on else "#2a2f2a",
        )

    # ---- actions -----------------------------------------------------------

    def toggle_effects(self):
        self.s.effects_on = not self.s.effects_on
        self._refresh_toggle()

    def set_color(self, bgr):
        self.s.color_bgr = bgr
        self.swatch.configure(bg=bgr_to_hex(bgr))

    def pick_color(self):
        rgb, _ = colorchooser.askcolor(color=bgr_to_hex(self.s.color_bgr), title="Glow color")
        if rgb:
            r, g, b = (int(v) for v in rgb)
            self.set_color((b, g, r))

    def toggle_fullscreen(self, value=None):
        self.fullscreen = (not self.fullscreen) if value is None else value
        self.root.attributes("-fullscreen", self.fullscreen)
        if self.fullscreen:
            self.panel.pack_forget()
        else:
            self.panel.pack(side="right", fill="y", before=self.video)

    def close(self):
        self.engine.stop()
        self.root.destroy()

    # ---- display loop ------------------------------------------------------

    def _tick(self):
        frame = self.engine.get_frame()
        if frame is not None:
            lw, lh = self.video.winfo_width(), self.video.winfo_height()
            fh, fw = frame.shape[:2]
            scale = min(lw / fw, lh / fh) if lw > 1 and lh > 1 else 1.0
            if abs(scale - 1.0) > 0.01:
                frame = cv2.resize(frame, (max(1, int(fw * scale)), max(1, int(fh * scale))),
                                   interpolation=cv2.INTER_LINEAR)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self._photo = ImageTk.PhotoImage(img)
            self.video.configure(image=self._photo)
        msg = self.engine.status or f"{self.engine.fps:4.1f} FPS"
        self.status.configure(text=msg)
        self.root.after(10, self._tick)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()
