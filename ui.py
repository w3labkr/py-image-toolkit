import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinterdnd2 import DND_FILES, TkinterDnD


def add_files(listbox):
    paths = filedialog.askopenfilenames()
    for path in paths:
        if path and path not in listbox.get(0, tk.END):
            listbox.insert(tk.END, path)


def browse_output(entry):
    path = filedialog.askdirectory()
    if path:
        entry.delete(0, tk.END)
        entry.insert(0, path)


def run_operation(listbox, output_entry, option_var):
    files = listbox.get(0, tk.END)
    output_dir = output_entry.get() or "output"
    if not files:
        messagebox.showerror("Error", "Select input files")
        return
    ops = {
        "Resize": ["python3", "resize.py"],
        "Crop": ["python3", "crop.py"],
        "OCR": ["python3", "ocr.py"],
        "Optimize": ["python3", "optimize.py"],
    }
    cmd_base = ops.get(option_var.get())
    if not cmd_base:
        messagebox.showerror("Error", "Select an operation")
        return
    successes = 0
    for f in files:
        cmd = cmd_base + [f, "-o", output_dir]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                successes += 1
            else:
                output = result.stderr.strip() or result.stdout.strip() or "Unknown error"
                messagebox.showerror("Error", f"{f}: {output}")
        except Exception as e:
            messagebox.showerror("Error", f"{f}: {e}")
    if successes:
        messagebox.showinfo("Success", f"{successes} file(s) processed")


def on_drop(event, listbox):
    for path in listbox.tk.splitlist(event.data):
        if path and path not in listbox.get(0, tk.END):
            listbox.insert(tk.END, path)


def main():
    root = TkinterDnD.Tk()
    root.title("Py Image Toolkit")

    style = ttk.Style(root)
    style.theme_use("clam")

    frame = ttk.Frame(root, padding=10)
    frame.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    ttk.Label(frame, text="Input Files").grid(row=0, column=0, sticky="w")
    file_list = tk.Listbox(frame, selectmode=tk.EXTENDED, width=50, height=5)
    file_list.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=5)
    file_list.drop_target_register(DND_FILES)
    file_list.dnd_bind("<<Drop>>", lambda e: on_drop(e, file_list))

    ttk.Button(frame, text="Add Files", command=lambda: add_files(file_list)).grid(row=2, column=0, sticky="w", pady=5)
    ttk.Button(frame, text="Clear", command=lambda: file_list.delete(0, tk.END)).grid(row=2, column=1, sticky="e", pady=5)

    ttk.Label(frame, text="Output Directory").grid(row=3, column=0, sticky="w")
    output_entry = ttk.Entry(frame, width=40)
    output_entry.grid(row=4, column=0, sticky="ew", pady=5)
    ttk.Button(frame, text="Browse", command=lambda: browse_output(output_entry)).grid(row=4, column=1, padx=5, pady=5)

    option_var = tk.StringVar(value="Resize")
    ttk.Label(frame, text="Operation").grid(row=5, column=0, sticky="w")
    ttk.OptionMenu(frame, option_var, "Resize", "Resize", "Crop", "OCR", "Optimize").grid(row=5, column=1, sticky="ew", pady=5)

    ttk.Button(
        frame,
        text="Run",
        command=lambda: run_operation(file_list, output_entry, option_var),
    ).grid(row=6, column=0, columnspan=2, pady=10)

    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(1, weight=0)

    root.mainloop()


if __name__ == "__main__":
    main()
