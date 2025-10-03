import sys
import json
from pathlib import Path
from typing import List, Dict

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
CLASSES = ["Helmet", "Vest", "Gloves", "Boots", "Mask", "Goggles", "Ear Protection"]

def discover_images(root: Path) -> List[Path]:
    return [p for p in sorted(root.rglob("*")) if p.suffix.lower() in SUPPORTED]

def scale_image(img: Image.Image, max_w=960, max_h=540) -> Image.Image:
    w, h = img.size
    scale = min(max_w / w, max_h / h, 1.0)
    return img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

class App:
    def __init__(self, master, images_dir: Path, json_path: Path):
        self.master = master
        self.images_dir = images_dir
        self.json_path = json_path

        # files
        self.images: List[Path] = discover_images(images_dir)
        if not self.images:
            messagebox.showerror("No images", f"No images found in {images_dir}")
            master.destroy()
            return

        # annotations: {filename: [labels]}
        self.annotations: Dict[str, List[str]] = {}
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.annotations = {k: (v if isinstance(v, list) else [str(v)]) for k, v in data.items()}
            except Exception as e:
                messagebox.showwarning("Warning", f"Could not read {json_path}:\n{e}")

        # order & navigation
        self.show_unlabeled_only = False
        self.order: List[int] = list(range(len(self.images)))  # list of indices into self.images
        self.idx_in_order = 0  # pointer into self.order

        # ui bits
        self.tk_img = None
        self.buttons: Dict[str, tk.Button] = {}

        # window
        master.title("Multi-label Image Annotator")
        master.geometry("1040x820")

        # top bar: path + counter
        top = tk.Frame(master); top.pack(fill=tk.X, padx=10, pady=8)
        self.path_label = tk.Label(top, text="", anchor="w")
        self.path_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.counter_label = tk.Label(top, text="")
        self.counter_label.pack(side=tk.RIGHT)

        # image canvas
        self.canvas = tk.Canvas(master, bg="#222222", width=960, height=540, highlightthickness=0)
        self.canvas.pack(padx=10, pady=10, expand=True)

        # label toggle row
        label_frame = tk.Frame(master); label_frame.pack(fill=tk.X, padx=10, pady=(0,6))
        for i, cls in enumerate(CLASSES, start=1):
            btn = tk.Button(label_frame, text=f"{i} - {cls}", width=12,
                            relief=tk.RAISED, bg="red", fg="white",
                            command=lambda c=cls: self.toggle_label(c))
            btn.pack(side=tk.LEFT, padx=5)
            self.buttons[cls] = btn

        clear_btn = tk.Button(label_frame, text="Clear All", width=12, bg="gray", fg="white",
                              command=self.clear_all_labels)
        clear_btn.pack(side=tk.LEFT, padx=10)

        copy_btn = tk.Button(label_frame, text="Copy Prev (U)", width=14,
                             command=self.copy_prev_labels)
        copy_btn.pack(side=tk.LEFT, padx=5)

        # current labels preview
        self.current_labels_text = tk.StringVar(value="Current labels: (none)")
        self.preview_label = tk.Label(master, textvariable=self.current_labels_text,
                                      anchor="w", fg="blue", font=("Arial", 12, "bold"))
        self.preview_label.pack(fill=tk.X, padx=10, pady=(2, 8))

        # progress row
        prog_frame = tk.Frame(master); prog_frame.pack(fill=tk.X, padx=10, pady=(0,8))
        self.prog = ttk.Progressbar(prog_frame, orient="horizontal", mode="determinate", length=720)
        self.prog.pack(side=tk.LEFT, padx=(0,10), fill=tk.X, expand=True)
        self.progress_text = tk.Label(prog_frame, text="")
        self.progress_text.pack(side=tk.LEFT)

        # navigation row
        nav = tk.Frame(master); nav.pack(fill=tk.X, padx=10, pady=6)
        self.back_btn = tk.Button(nav, text="⟵ Back", command=self.go_back); self.back_btn.pack(side=tk.LEFT)
        self.save_btn = tk.Button(nav, text="💾 Save (S)", command=self.save_current); self.save_btn.pack(side=tk.LEFT, padx=6)
        self.next_btn = tk.Button(nav, text="Next ⟶", command=self.go_next); self.next_btn.pack(side=tk.LEFT)

        # jump + filter
        tk.Label(nav, text="Jump to #").pack(side=tk.LEFT, padx=(16,4))
        self.jump_entry = tk.Entry(nav, width=6)
        self.jump_entry.pack(side=tk.LEFT)
        tk.Button(nav, text="Go", command=self.jump_to_index).pack(side=tk.LEFT, padx=(4,12))

        self.unlabeled_var = tk.IntVar(value=0)
        self.unlabeled_chk = tk.Checkbutton(nav, text="Unlabeled Only", variable=self.unlabeled_var,
                                            command=self.toggle_unlabeled_only)
        self.unlabeled_chk.pack(side=tk.LEFT)

        # status line
        self.status = tk.StringVar(value="Ready")
        tk.Label(master, textvariable=self.status, anchor="w").pack(fill=tk.X, padx=10, pady=(4, 10))

        # hotkeys
        master.bind("<Right>", lambda e: self.go_next())
        master.bind("<Left>", lambda e: self.go_back())
        master.bind("1", lambda e: self.toggle_label("Helmet"))
        master.bind("2", lambda e: self.toggle_label("Vest"))
        master.bind("3", lambda e: self.toggle_label("Gloves"))
        master.bind("4", lambda e: self.toggle_label("Boots"))
        master.bind("5", lambda e: self.toggle_label("Mask"))
        master.bind("6", lambda e: self.toggle_label("Goggles"))
        master.bind("7", lambda e: self.toggle_label("Ear Protection"))
        master.bind("s", lambda e: self.save_current())
        master.bind("S", lambda e: self.save_current())
        master.bind("c", lambda e: self.clear_all_labels())
        master.bind("C", lambda e: self.clear_all_labels())
        master.bind("u", lambda e: self.copy_prev_labels())
        master.bind("U", lambda e: self.copy_prev_labels())
        master.bind("g", lambda e: self.jump_entry.focus_set())
        master.bind("G", lambda e: self.jump_entry.focus_set())

        # show first
        self.show_image()

    # ---------- helpers ----------
    def current_abs_index(self) -> int:
        return self.order[self.idx_in_order]

    def current_path(self) -> Path:
        return self.images[self.current_abs_index()]

    def key(self) -> str:
        return self.current_path().name  # filename as key

    def recalc_order(self):
        """Recompute browsing order depending on unlabeled-only toggle."""
        if self.show_unlabeled_only:
            self.order = [i for i, p in enumerate(self.images)
                          if not self.annotations.get(self.images[i].name)]
        else:
            self.order = list(range(len(self.images)))
        # keep pointer in range
        self.idx_in_order = max(0, min(self.idx_in_order, len(self.order)-1))

    def count_labeled(self) -> int:
        return sum(1 for p in self.images if self.annotations.get(p.name))

    def update_progress(self):
        labeled = self.count_labeled()
        total = len(self.images)
        self.prog["maximum"] = total
        self.prog["value"] = labeled
        self.progress_text.config(text=f"{labeled}/{total} labeled")

    def set_button_state_from_labels(self, labels: List[str]):
        s = set(labels)
        for cls, btn in self.buttons.items():
            if cls in s:
                btn.config(bg="green", relief=tk.SUNKEN)
            else:
                btn.config(bg="red", relief=tk.RAISED)

    def update_preview(self):
        labels = self.annotations.get(self.key(), [])
        self.current_labels_text.set("Current labels: " + (", ".join(labels) if labels else "(none)"))

    # ---------- UI actions ----------
    def show_image(self):
        if not self.order:
            # nothing to show (e.g., unlabeled-only and none left)
            self.canvas.delete("all")
            self.path_label.config(text="(no images match current filter)")
            self.counter_label.config(text="")
            self.set_button_state_from_labels([])
            self.update_preview()
            self.update_progress()
            self.status.set("No images to show for current filter.")
            return

        p = self.current_path()
        self.path_label.config(text=str(p))
        self.counter_label.config(text=f"{self.idx_in_order+1}/{len(self.order)} (filtered view)")

        try:
            img = Image.open(p).convert("RGB")
            img = scale_image(img)
            self.tk_img = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            cw, ch = 960, 540
            iw, ih = img.size
            x, y = (cw - iw) // 2, (ch - ih) // 2
            self.canvas.create_image(x, y, anchor="nw", image=self.tk_img)
        except Exception as e:
            self.canvas.delete("all")
            self.status.set(f"Failed to load image: {e}")

        self.set_button_state_from_labels(self.annotations.get(self.key(), []))
        self.update_preview()
        self.update_progress()
        self.status.set("Ready")

    def toggle_label(self, cls: str):
        labels = set(self.annotations.get(self.key(), []))
        if cls in labels:
            labels.remove(cls)
        else:
            labels.add(cls)
        self.annotations[self.key()] = sorted(labels, key=lambda c: CLASSES.index(c))
        self.set_button_state_from_labels(self.annotations[self.key()])
        self.update_preview()
        self.status.set(f"Toggled {cls}")

    def clear_all_labels(self):
        self.annotations[self.key()] = []
        self.set_button_state_from_labels([])
        self.update_preview()
        self.status.set("Cleared all labels")

    def copy_prev_labels(self):
        if self.idx_in_order > 0:
            prev_abs = self.order[self.idx_in_order - 1]
            prev_key = self.images[prev_abs].name
            prev_labels = list(self.annotations.get(prev_key, []))
            self.annotations[self.key()] = prev_labels
            self.set_button_state_from_labels(prev_labels)
            self.update_preview()
            self.status.set(f"Copied labels from previous: {prev_labels}")
        else:
            self.status.set("No previous image to copy from.")

    def save_current(self):
        try:
            self.json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(self.annotations, f, indent=2, ensure_ascii=False)
            self.status.set(f"Saved {self.key()} → {self.annotations.get(self.key(), [])}")
            self.update_progress()
        except Exception as e:
            messagebox.showerror("Save error", f"Could not save JSON:\n{e}")

    def go_next(self):
        self.save_current()
        if not self.order:
            self.status.set("No images to navigate.")
            return
        if self.idx_in_order < len(self.order) - 1:
            self.idx_in_order += 1
            self.show_image()
        else:
            self.status.set("You’re at the last image 🏁")

    def go_back(self):
        self.save_current()
        if not self.order:
            self.status.set("No images to navigate.")
            return
        if self.idx_in_order > 0:
            self.idx_in_order -= 1
            self.show_image()
        else:
            self.status.set("You’re at the first image ⏮️")

    def jump_to_index(self):
        try:
            # user enters 1-based index within current filtered list
            i = int(self.jump_entry.get()) - 1
        except Exception:
            self.status.set("Enter a number to jump."); return
        if not self.order:
            self.status.set("No images to navigate.")
            return
        i = max(0, min(i, len(self.order)-1))
        self.idx_in_order = i
        self.show_image()

    def toggle_unlabeled_only(self):
        self.show_unlabeled_only = bool(self.unlabeled_var.get())
        prev_abs = self.current_abs_index() if self.order else 0
        self.recalc_order()
        # keep current image in view if possible
        if self.order:
            if prev_abs in self.order:
                self.idx_in_order = self.order.index(prev_abs)
            else:
                self.idx_in_order = 0
        self.show_image()

def main():
    images_dir = Path("dataset2/train/images")  # Hardcoded path
    json_path = Path("dataset2/train/annotations.json")  # Hardcoded JSON output path

    root = tk.Tk()
    root.lift(); root.attributes("-topmost", True); root.after(600, lambda: root.attributes("-topmost", False))
    App(root, images_dir, json_path)
    root.mainloop()

if __name__ == "__main__":
    main()