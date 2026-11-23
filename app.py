"""
LIVE DEMO: draw a digit with the mouse and the model predicts it.
Run:  python app.py
(Needs the trained model -> run tf_cnn.py first.)
"""
import tkinter as tk
from PIL import Image, ImageDraw
from common import load_model, preprocess

CANVAS = 300
BRUSH = 20


class App:
    def __init__(self, root):
        self.root = root
        root.title("Handwritten Digit Recognition - Live Demo")
        root.configure(bg="#1e1e2e")
        self.model = load_model()
        self.model.predict(preprocess_dummy(), verbose=0)   # warm-up

        left = tk.Frame(root, bg="#1e1e2e"); left.grid(row=0, column=0, padx=15, pady=15)
        right = tk.Frame(root, bg="#1e1e2e"); right.grid(row=0, column=1, padx=15, pady=15)

        tk.Label(left, text="Draw a digit (0-9)", fg="white", bg="#1e1e2e",
                 font=("Arial", 14, "bold")).pack(pady=(0, 8))
        self.canvas = tk.Canvas(left, width=CANVAS, height=CANVAS, bg="black",
                                highlightthickness=2, highlightbackground="#89b4fa",
                                cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonPress-1>", self.start)
        self.canvas.bind("<ButtonRelease-1>", lambda e: self.predict())

        tk.Button(left, text="Clear", width=14, font=("Arial", 12),
                  command=self.clear).pack(pady=10)

        self.result = tk.Label(right, text="?", fg="#a6e3a1", bg="#1e1e2e",
                               font=("Arial", 110, "bold"), width=2)
        self.result.pack()
        self.conf = tk.Label(right, text="Confidence: --", fg="white", bg="#1e1e2e",
                             font=("Arial", 14))
        self.conf.pack(pady=(0, 10))

        self.bars = []
        for d in range(10):
            row = tk.Frame(right, bg="#1e1e2e"); row.pack(anchor="w", pady=1)
            tk.Label(row, text=str(d), fg="white", bg="#1e1e2e", width=2,
                     font=("Arial", 11)).pack(side="left")
            c = tk.Canvas(row, width=200, height=14, bg="#313244", highlightthickness=0)
            c.pack(side="left")
            rect = c.create_rectangle(0, 0, 0, 14, fill="#89b4fa", width=0)
            pct = tk.Label(row, text="0%", fg="white", bg="#1e1e2e", width=5,
                           font=("Arial", 10))
            pct.pack(side="left")
            self.bars.append((c, rect, pct))

        self.clear()

    def start(self, e):
        self.last = (e.x, e.y)
        self.draw(e)

    def draw(self, e):
        x0, y0 = self.last
        self.canvas.create_line(x0, y0, e.x, e.y, fill="white", width=BRUSH,
                                capstyle=tk.ROUND, smooth=True)
        self.pdraw.line([x0, y0, e.x, e.y], fill=255, width=BRUSH)
        r = BRUSH // 2
        self.pdraw.ellipse([e.x - r, e.y - r, e.x + r, e.y + r], fill=255)
        self.last = (e.x, e.y)

    def predict(self):
        x = preprocess(self.img)
        if x is None:
            return
        probs = self.model.predict(x, verbose=0)[0]
        best = int(probs.argmax())
        self.result.config(text=str(best))
        self.conf.config(text=f"Confidence: {probs[best] * 100:.1f}%")
        for d, (c, rect, pct) in enumerate(self.bars):
            c.coords(rect, 0, 0, 200 * float(probs[d]), 14)
            c.itemconfig(rect, fill="#a6e3a1" if d == best else "#89b4fa")
            pct.config(text=f"{probs[d] * 100:.0f}%")

    def clear(self):
        self.canvas.delete("all")
        self.img = Image.new("L", (CANVAS, CANVAS), 0)
        self.pdraw = ImageDraw.Draw(self.img)
        self.result.config(text="?")
        self.conf.config(text="Confidence: --")
        for c, rect, pct in self.bars:
            c.coords(rect, 0, 0, 0, 14)
            c.itemconfig(rect, fill="#89b4fa")
            pct.config(text="0%")


def preprocess_dummy():
    import numpy as np
    return np.zeros((1, 28, 28, 1), dtype="float32")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
