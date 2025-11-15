# dataset_summary.py

import os, json
import tkinter as tk
from tkinter import ttk, filedialog

class DatasetSummaryApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dataset Summary")
        self.geometry("600x400")
        self.configure(bg="white")

        self.folder = None

        top = ttk.Frame(self); top.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(top, text="Select folder…", command=self.choose_folder).pack(side=tk.LEFT)
        self.lbl_folder = ttk.Label(top, text="No folder selected"); self.lbl_folder.pack(side=tk.LEFT, padx=10)

        self.txt = tk.Text(self, bg="white", fg="black")
        self.txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def choose_folder(self):
        folder = filedialog.askdirectory(title="Select graph JSON folder")
        if not folder:
            return
        self.folder = folder
        self.lbl_folder.config(text=folder)
        self.summarize()

    def summarize(self):
        files = [f for f in os.listdir(self.folder) if f.endswith(".json")]
        num_files = len(files)
        node_counts = []
        room_types = {}

        for fname in files:
            path = os.path.join(self.folder, fname)
            with open(path, "r", encoding="utf-8") as f:
                g = json.load(f)
            nodes = g.get("nodes", [])
            node_counts.append(len(nodes))
            for n in nodes:
                rt = n.get("room_type")
                if rt:
                    room_types[rt] = room_types.get(rt, 0) + 1

        self.txt.delete("1.0", tk.END)
        if not node_counts:
            self.txt.insert(tk.END, "No JSON files with nodes found.\n")
            return

        self.txt.insert(tk.END, f"Folder: {self.folder}\n")
        self.txt.insert(tk.END, f"Number of files: {num_files}\n")
        self.txt.insert(tk.END, f"Min nodes: {min(node_counts)}\n")
        self.txt.insert(tk.END, f"Max nodes: {max(node_counts)}\n")
        avg = sum(node_counts)/len(node_counts)
        self.txt.insert(tk.END, f"Average nodes: {avg:.1f}\n\n")

        self.txt.insert(tk.END, "Room types (for padding/masking):\n")
        for rt, count in sorted(room_types.items(), key=lambda x: -x[1]):
            self.txt.insert(tk.END, f"  {rt}: {count}\n")

if __name__ == "__main__":
    DatasetSummaryApp().mainloop()
