import os
import sys
import json
import tempfile
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CLEAN_UP_SCRIPT = os.path.join(REPO_ROOT, "clean_data", "clean_up.py")
OUTPUT_BASE = os.path.join("data", "processed")  # 暫時固定，未來開放


def run_clean_up_tasks(tasks, status_var, on_done):
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False)

    def worker():
        try:
            result = subprocess.run(
                [sys.executable, CLEAN_UP_SCRIPT, "--tasks-json", tmp],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if result.returncode == 0:
                status_var.set("資料清洗完成")
                messagebox.showinfo("資料清洗完成", result.stdout or "處理完成")
            else:
                status_var.set("資料清洗失敗")
                messagebox.showerror(
                    "資料清洗失敗",
                    f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}",
                )
        except Exception as e:
            status_var.set("資料清洗錯誤")
            messagebox.showerror("資料清洗錯誤", str(e))
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            on_done()

    threading.Thread(target=worker, daemon=True).start()


def open_clean_dialog(parent, status_var, main_btn):
    dialog = tk.Toplevel(parent)
    dialog.title("資料清洗 - 選擇 Excel")
    dialog.geometry("720x480")
    dialog.transient(parent)
    dialog.grab_set()

    # 輸出位置（唯讀顯示）
    top = tk.Frame(dialog)
    top.pack(fill=tk.X, padx=12, pady=(12, 4))
    tk.Label(top, text="輸出位置：").pack(side=tk.LEFT)
    out_var = tk.StringVar(value=OUTPUT_BASE)
    out_entry = tk.Entry(
        top,
        textvariable=out_var,
        state="readonly",
        width=60,
        readonlybackground="#f0f0f0",
    )
    out_entry.pack(side=tk.LEFT, padx=4)

    # 選檔列
    pick_frame = tk.Frame(dialog)
    pick_frame.pack(fill=tk.X, padx=12, pady=6)
    count_var = tk.StringVar(value="尚未選擇")

    task_state = []  # [(excel_path, name_var), ...]

    # 任務清單區（含滾動）
    list_box = tk.LabelFrame(dialog, text="待處理清單")
    list_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

    canvas = tk.Canvas(list_box, borderwidth=0, highlightthickness=0)
    scrollbar = tk.Scrollbar(list_box, orient="vertical", command=canvas.yview)
    inner = tk.Frame(canvas)
    inner.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def render():
        for child in inner.winfo_children():
            child.destroy()
        if not task_state:
            tk.Label(inner, text="(尚未選擇檔案)", fg="gray").grid(
                row=0, column=0, padx=8, pady=8
            )
            return
        tk.Label(inner, text="Excel 檔案", font=("Arial", 9, "bold")).grid(
            row=0, column=0, sticky="w", padx=4, pady=4
        )
        tk.Label(inner, text="輸出資料夾名稱", font=("Arial", 9, "bold")).grid(
            row=0, column=1, sticky="w", padx=4, pady=4
        )
        tk.Label(inner, text="", width=4).grid(row=0, column=2)
        for i, (path, name_var) in enumerate(task_state, start=1):
            tk.Label(inner, text=os.path.basename(path), anchor="w").grid(
                row=i, column=0, sticky="w", padx=4, pady=2
            )
            tk.Entry(inner, textvariable=name_var, width=32).grid(
                row=i, column=1, padx=4, pady=2
            )

            def make_remove(idx):
                return lambda: remove_at(idx)

            tk.Button(inner, text="✕", width=2, command=make_remove(i - 1)).grid(
                row=i, column=2, padx=2
            )

    def remove_at(idx):
        if 0 <= idx < len(task_state):
            task_state.pop(idx)
            update_count()
            render()

    def update_count():
        if task_state:
            count_var.set(f"共 {len(task_state)} 個檔案")
        else:
            count_var.set("尚未選擇")

    def pick():
        initial = os.path.join(REPO_ROOT, "data", "raw")
        if not os.path.isdir(initial):
            initial = REPO_ROOT
        files = filedialog.askopenfilenames(
            parent=dialog,
            title="選擇 Excel 檔案（可多選）",
            initialdir=initial,
            filetypes=[("Excel files", "*.xlsx *.xlsm"), ("All files", "*.*")],
        )
        if not files:
            return
        existing = {p for p, _ in task_state}
        for f in files:
            if f in existing:
                continue
            base = os.path.splitext(os.path.basename(f))[0]
            task_state.append((f, tk.StringVar(value=base)))
        update_count()
        render()

    def clear():
        task_state.clear()
        update_count()
        render()

    def execute():
        if not task_state:
            messagebox.showwarning("提示", "請先選擇 Excel 檔案", parent=dialog)
            return
        seen_names = {}
        tasks = []
        for path, name_var in task_state:
            name = name_var.get().strip()
            if not name:
                messagebox.showwarning(
                    "提示",
                    f"{os.path.basename(path)} 的輸出資料夾名稱不可為空",
                    parent=dialog,
                )
                return
            if name in seen_names:
                messagebox.showwarning(
                    "提示",
                    f"輸出資料夾名稱重複：{name}\n({os.path.basename(seen_names[name])} 與 {os.path.basename(path)})",
                    parent=dialog,
                )
                return
            seen_names[name] = path
            tasks.append({"excel": path, "output_name": name})

        exec_btn.config(state=tk.DISABLED)
        cancel_btn.config(state=tk.DISABLED)
        pick_btn.config(state=tk.DISABLED)
        clear_btn.config(state=tk.DISABLED)
        status_var.set("資料清洗執行中...")

        def done():
            dialog.destroy()
            main_btn.config(state=tk.NORMAL)

        run_clean_up_tasks(tasks, status_var, done)

    pick_btn = tk.Button(pick_frame, text="選擇 Excel 檔案（可多選）", command=pick)
    pick_btn.pack(side=tk.LEFT)
    clear_btn = tk.Button(pick_frame, text="清空", command=clear)
    clear_btn.pack(side=tk.LEFT, padx=4)
    tk.Label(pick_frame, textvariable=count_var, fg="gray").pack(side=tk.LEFT, padx=8)

    bot = tk.Frame(dialog)
    bot.pack(fill=tk.X, padx=12, pady=10)
    exec_btn = tk.Button(
        bot, text="執行清洗", command=execute, bg="#4a90e2", fg="white", width=12
    )
    exec_btn.pack(side=tk.RIGHT, padx=4)

    def on_cancel():
        dialog.destroy()
        main_btn.config(state=tk.NORMAL)

    cancel_btn = tk.Button(bot, text="取消", command=on_cancel, width=10)
    cancel_btn.pack(side=tk.RIGHT)

    render()
    dialog.protocol("WM_DELETE_WINDOW", on_cancel)


def on_clean_click(root, status_var, btn):
    btn.config(state=tk.DISABLED)
    open_clean_dialog(root, status_var, btn)


def main():
    root = tk.Tk()
    root.title("Probe-Mark")
    root.geometry("420x260")
    root.resizable(False, False)

    title = tk.Label(root, text="Probe-Mark", font=("Arial", 18, "bold"))
    title.pack(pady=(18, 6))

    subtitle = tk.Label(root, text="探針痕跡影像辨識系統", font=("Arial", 10))
    subtitle.pack(pady=(0, 14))

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=4)

    status_var = tk.StringVar(value="")

    btn_clean = tk.Button(
        btn_frame, text="資料清洗", width=10, height=2, font=("Arial", 10)
    )
    btn_clean.config(command=lambda: on_clean_click(root, status_var, btn_clean))
    btn_clean.grid(row=0, column=0, padx=6)

    btn_train = tk.Button(
        btn_frame,
        text="模型訓練",
        width=10,
        height=2,
        font=("Arial", 10),
        state=tk.DISABLED,
    )
    btn_train.grid(row=0, column=1, padx=6)

    btn_pred = tk.Button(
        btn_frame,
        text="模型預測",
        width=10,
        height=2,
        font=("Arial", 10),
        state=tk.DISABLED,
    )
    btn_pred.grid(row=0, column=2, padx=6)

    status_label = tk.Label(root, textvariable=status_var, fg="blue")
    status_label.pack(pady=16)

    root.mainloop()


if __name__ == "__main__":
    main()
