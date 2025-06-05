import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox


def browse_input(entry):
    path = filedialog.askopenfilename()
    if path:
        entry.delete(0, tk.END)
        entry.insert(0, path)


def browse_output(entry):
    path = filedialog.askdirectory()
    if path:
        entry.delete(0, tk.END)
        entry.insert(0, path)


def run_operation(input_entry, output_entry, option_var):
    input_path = input_entry.get()
    output_dir = output_entry.get() or "output"
    if not input_path:
        messagebox.showerror("Error", "Select an input file")
        return
    ops = {
        "Resize": ["python3", "resize.py"],
        "Crop": ["python3", "crop.py"],
        "OCR": ["python3", "ocr.py"],
        "Optimize": ["python3", "optimize.py"],
    }
    cmd = ops.get(option_var.get())
    if not cmd:
        messagebox.showerror("Error", "Select an operation")
        return
    cmd += [input_path, "-o", output_dir]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            messagebox.showinfo("Success", f"{option_var.get()} completed")
        else:
            output = result.stderr.strip() or result.stdout.strip() or "Unknown error"
            messagebox.showerror("Error", output)
    except Exception as e:
        messagebox.showerror("Error", str(e))


def main():
    root = tk.Tk()
    root.title("Py Image Toolkit UI")

    tk.Label(root, text="Input File").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    input_entry = tk.Entry(root, width=40)
    input_entry.grid(row=0, column=1, padx=5, pady=5)
    tk.Button(root, text="Browse", command=lambda: browse_input(input_entry)).grid(row=0, column=2, padx=5, pady=5)

    tk.Label(root, text="Output Directory").grid(row=1, column=0, sticky="w", padx=5, pady=5)
    output_entry = tk.Entry(root, width=40)
    output_entry.grid(row=1, column=1, padx=5, pady=5)
    tk.Button(root, text="Browse", command=lambda: browse_output(output_entry)).grid(row=1, column=2, padx=5, pady=5)

    option_var = tk.StringVar(value="Resize")
    tk.Label(root, text="Operation").grid(row=2, column=0, sticky="w", padx=5, pady=5)
    tk.OptionMenu(root, option_var, "Resize", "Crop", "OCR", "Optimize").grid(row=2, column=1, padx=5, pady=5, sticky="w")

    tk.Button(
        root,
        text="Run",
        command=lambda: run_operation(input_entry, output_entry, option_var),
    ).grid(row=3, column=0, columnspan=3, pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
