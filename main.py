import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from mpivr20_cms import get_clean_output_dir

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = get_clean_output_dir()
EXCEL_EXTENSIONS = (".xlsx", ".xlsm")
EXCEL_FILETYPES = [("Excel files", "*.xlsx *.xlsm"), ("All files", "*.*")]


def run_background(status_var, done_message, target):
    def worker():
        try:
            status_var.set("執行中...")
            result = target()
            status_var.set(done_message)
            messagebox.showinfo(done_message, str(result) if result is not None else done_message)
        except Exception as exc:
            status_var.set("執行失敗")
            messagebox.showerror("執行失敗", f"{type(exc).__name__}: {exc}")

    threading.Thread(target=worker, daemon=True).start()


def choose_file(var, filetypes):
    path = filedialog.askopenfilename(filetypes=filetypes)
    if path:
        var.set(path)


def choose_files(callback):
    files = filedialog.askopenfilenames(
        initialdir=os.path.join(REPO_ROOT, "data", "raw"),
        filetypes=EXCEL_FILETYPES,
    )
    if files:
        callback(files)


def default_dataset_name(_folder: str, excel_path: str) -> str:
    """Build a dataset name from an Excel path selected through folder import."""
    return os.path.splitext(os.path.basename(excel_path))[0]


def scan_excel_folder(folder: str) -> list[dict[str, str]]:
    """Collect supported Excel files directly under a selected folder."""
    tasks = []
    for filename in sorted(os.listdir(folder)):
        excel_path = os.path.join(folder, filename)
        if not os.path.isfile(excel_path):
            continue
        if filename.startswith("~$"):
            continue
        if not filename.lower().endswith(EXCEL_EXTENSIONS):
            continue
        tasks.append(
            {
                "excel": excel_path,
                "name": default_dataset_name(folder, excel_path),
                "display": filename,
            }
        )
    return tasks


def choose_excel_folder(callback) -> None:
    """Prompt for a folder and send scanned Excel tasks to the callback."""
    folder = filedialog.askdirectory(initialdir=os.path.join(REPO_ROOT, "data", "raw"))
    if folder:
        callback(scan_excel_folder(folder))


def format_clean_result(result: dict) -> str:
    """Build the final clean dialog message, including skipped Excel files."""
    lines = [
        f"成功 dataset ids: {result['dataset_ids']}",
        f"成功樣本數: {result['total_samples']}",
    ]
    failed = result.get("failed", [])
    if failed:
        lines.append(f"略過檔案: {len(failed)}")
        for item in failed:
            lines.append(f"- {item['excel']}: {item['error']}")
    return "\n".join(lines)


def choose_dir(var):
    path = filedialog.askdirectory()
    if path:
        var.set(path)


def open_clean_dialog(parent, status_var):
    dialog = tk.Toplevel(parent)
    dialog.title("資料清洗")
    dialog.geometry("760x520")
    dialog.transient(parent)
    dialog.grab_set()

    task_state = []
    force_var = tk.BooleanVar(value=False)

    top = ttk.Frame(dialog, padding=12)
    top.pack(fill=tk.X)
    ttk.Label(top, text=f"輸出位置：{OUTPUT_BASE}").pack(side=tk.LEFT)
    ttk.Checkbutton(top, text="強制重建快取", variable=force_var).pack(side=tk.RIGHT)

    list_box = ttk.LabelFrame(dialog, text="待處理 Excel", padding=8)
    list_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
    rows_canvas = tk.Canvas(list_box, highlightthickness=0)
    rows_scrollbar = ttk.Scrollbar(list_box, orient=tk.VERTICAL, command=rows_canvas.yview)
    rows_canvas.configure(yscrollcommand=rows_scrollbar.set)
    rows_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    rows_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    rows = ttk.Frame(rows_canvas)
    rows_window = rows_canvas.create_window((0, 0), window=rows, anchor="nw")

    def update_scroll_region(_event: tk.Event | None = None) -> None:
        """Keep the Excel task list scroll range aligned with rendered rows."""
        rows_canvas.configure(scrollregion=rows_canvas.bbox("all"))

    def fit_rows_width(event: tk.Event) -> None:
        """Stretch the row frame to match the visible canvas width."""
        rows_canvas.itemconfigure(rows_window, width=event.width)

    rows.bind("<Configure>", update_scroll_region)
    rows_canvas.bind("<Configure>", fit_rows_width)

    def render():
        for child in rows.winfo_children():
            child.destroy()
        if not task_state:
            ttk.Label(rows, text="尚未選擇 Excel").grid(row=0, column=0, sticky="w")
            return
        ttk.Label(rows, text="Excel").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Label(rows, text="dataset_name").grid(row=0, column=1, sticky="w", padx=4)
        for idx, item in enumerate(task_state, start=1):
            ttk.Label(rows, text=item.get("display") or os.path.basename(item["excel"])).grid(
                row=idx, column=0, sticky="w", padx=4, pady=2
            )
            ttk.Entry(rows, textvariable=item["name_var"], width=34).grid(
                row=idx, column=1, sticky="w", padx=4, pady=2
            )

    def add_task(excel: str, name: str, display: str | None = None) -> bool:
        """Add one Excel task to the dialog state if it is not already listed."""
        existing = {item["excel"] for item in task_state}
        if excel in existing:
            return False
        task_state.append(
            {
                "excel": excel,
                "name_var": tk.StringVar(value=name),
                "display": display,
            }
        )
        return True

    def add_files(files: tuple[str, ...]) -> None:
        """Add Excel files selected one by one."""
        added = False
        for path in files:
            base = os.path.splitext(os.path.basename(path))[0]
            added = add_task(path, base) or added
        if added:
            render()

    def add_folder_tasks(tasks: list[dict[str, str]]) -> None:
        """Add every Excel task discovered from a selected folder."""
        added = False
        for task in tasks:
            added = add_task(task["excel"], task["name"], task["display"]) or added
        if added:
            render()

    def clear():
        task_state.clear()
        render()

    def execute():
        tasks = []
        seen = set()
        for item in task_state:
            name = item["name_var"].get().strip()
            if not name:
                messagebox.showwarning("資料清洗", "dataset_name 不可空白", parent=dialog)
                return
            if name in seen:
                messagebox.showwarning("資料清洗", f"dataset_name 重複：{name}", parent=dialog)
                return
            seen.add(name)
            tasks.append({"excel": item["excel"], "output_name": name})
        if not tasks:
            messagebox.showwarning("資料清洗", "請先選擇 Excel", parent=dialog)
            return

        def task():
            from load_clean_module import LoadCleanService

            service = LoadCleanService(
                tasks=tasks,
                save_dir=OUTPUT_BASE,
                force=force_var.get(),
            )
            result = service.run()
            return format_clean_result(result)

        run_background(status_var, "資料清洗完成", task)
        dialog.destroy()

    actions = ttk.Frame(dialog, padding=12)
    actions.pack(fill=tk.X)
    ttk.Button(actions, text="選擇 Excel", command=lambda: choose_files(add_files)).pack(
        side=tk.LEFT
    )
    ttk.Button(
        actions, text="選擇資料夾", command=lambda: choose_excel_folder(add_folder_tasks)
    ).pack(side=tk.LEFT, padx=6)
    ttk.Button(actions, text="清空", command=clear).pack(side=tk.LEFT, padx=6)
    ttk.Button(actions, text="開始清洗", command=execute).pack(side=tk.RIGHT)
    ttk.Button(actions, text="取消", command=dialog.destroy).pack(side=tk.RIGHT, padx=6)

    render()


def dataset_options():
    from database import DBManager

    db = DBManager()
    db.init_db()
    datasets = db.list_datasets(only_done=True)
    labels = []
    mapping = {}
    for dataset in datasets:
        label = f"#{dataset['id']} {dataset['dataset_name']} ({dataset['n_samples']} samples)"
        labels.append(label)
        mapping[label] = dataset["id"]
    return labels, mapping


def model_options():
    from database import DBManager

    db = DBManager()
    db.init_db()
    models = db.list_models()
    labels = []
    mapping = {}
    for model in models:
        label = (
            f"#{model['id']} run#{model['run_id']} "
            f"{model['run_name'] or ''} {model['encoder_name']}+{model['decoder_name']}"
        )
        labels.append(label)
        mapping[label] = model["id"]
    return labels, mapping


def open_train_dialog(parent, status_var):
    dialog = tk.Toplevel(parent)
    dialog.title("模型訓練")
    dialog.geometry("680x520")
    dialog.transient(parent)
    dialog.grab_set()

    labels, mapping = dataset_options()
    dataset_var = tk.StringVar(value=labels[0] if labels else "")
    field_defs = [
        ("run_name", "Run 名稱", "probe_mark", "本次訓練的名稱；會影響 runs/ 底下的輸出資料夾。"),
        ("split_name", "Split 名稱", "default", "資料切分名稱；相同 dataset + split 名稱會覆寫舊切分。"),
        ("val_ratio", "驗證比例", "0.2", "從 dataset 中切出多少比例做 validation；0.2 表示 80% train / 20% val。"),
        ("encoder_name", "Encoder", "resnet18", "特徵抽取 backbone，例如 resnet18、resnet50；需為 segmentation_models_pytorch 支援名稱。"),
        ("encoder_weights", "Encoder 權重", "imagenet", "Encoder 預訓練權重；常用 imagenet，留空表示不載入預訓練權重。"),
        ("decoder_name", "Decoder", "FPN", "分割模型架構，例如 FPN、Unet、DeepLabV3Plus。"),
        ("max_epochs", "Epochs", "10", "完整看過訓練資料的次數；越大訓練越久。"),
        ("batch_size", "Batch size", "16", "每次送進模型的影像張數；GPU 記憶體不足時調小。"),
        ("num_workers", "Workers", "0", "DataLoader 背景讀圖程序數；Windows/Tk 介面建議先用 0。"),
        ("lr", "Learning rate", "0.0001", "Adam optimizer 學習率；太大可能不穩，太小會學得慢。"),
        ("eta_min", "最低 LR", "0.00001", "CosineAnnealingLR 的最低 learning rate。"),
        ("gpu_id", "GPU ID", "0", "使用哪張 GPU；-1 表示 CPU。"),
    ]
    fields = {key: tk.StringVar(value=default) for key, _label, default, _help in field_defs}

    body = ttk.Frame(dialog, padding=12)
    body.pack(fill=tk.BOTH, expand=True)
    body.columnconfigure(2, weight=1)
    ttk.Label(body, text="Dataset").grid(row=0, column=0, sticky="e", pady=4, padx=4)
    combo = ttk.Combobox(body, textvariable=dataset_var, values=labels, state="readonly", width=44)
    combo.grid(row=0, column=2, sticky="w", pady=4, padx=4)
    ttk.Button(
        body,
        text="?",
        width=3,
        command=lambda: messagebox.showinfo(
            "Dataset",
            "選擇已完成資料清洗並登記在 DB 的 dataset。",
            parent=dialog,
        ),
    ).grid(row=0, column=1, sticky="w", pady=4, padx=4)
    if not labels:
        ttk.Label(body, text="尚無 dataset，請先執行資料清洗").grid(
            row=1, column=2, sticky="w", pady=4
        )

    for row, (key, label, _default, help_text) in enumerate(field_defs, start=1):
        ttk.Label(body, text=label).grid(row=row, column=0, sticky="e", pady=3, padx=4)
        ttk.Entry(body, textvariable=fields[key], width=32).grid(
            row=row, column=2, sticky="w", pady=3
        )
        ttk.Button(
            body,
            text="?",
            width=3,
            command=lambda title=label, text=help_text: messagebox.showinfo(
                title,
                text,
                parent=dialog,
            ),
        ).grid(row=row, column=1, sticky="w", pady=3, padx=4)

    def execute():
        dataset_id = mapping.get(dataset_var.get())
        if dataset_id is None:
            messagebox.showwarning("模型訓練", "請先選擇 dataset", parent=dialog)
            return

        def task():
            from train_module import Trainer

            return Trainer(
                dataset_id=dataset_id,
                split_name=fields["split_name"].get().strip() or "default",
                run_name=fields["run_name"].get().strip() or None,
                val_ratio=float(fields["val_ratio"].get()),
                encoder_name=fields["encoder_name"].get().strip(),
                encoder_weights=fields["encoder_weights"].get().strip() or None,
                decoder_name=fields["decoder_name"].get().strip(),
                max_epochs=int(fields["max_epochs"].get()),
                batch_size=int(fields["batch_size"].get()),
                num_workers=int(fields["num_workers"].get()),
                lr=float(fields["lr"].get()),
                eta_min=float(fields["eta_min"].get()),
                gpu_id=int(fields["gpu_id"].get()),
            ).run()

        run_background(status_var, "模型訓練完成", task)
        dialog.destroy()

    actions = ttk.Frame(dialog, padding=12)
    actions.pack(fill=tk.X)
    ttk.Button(actions, text="開始訓練", command=execute).pack(side=tk.RIGHT)
    ttk.Button(actions, text="取消", command=dialog.destroy).pack(side=tk.RIGHT, padx=6)


def open_predict_dialog(parent, status_var):
    dialog = tk.Toplevel(parent)
    dialog.title("模型預測")
    dialog.geometry("660x260")
    dialog.transient(parent)
    dialog.grab_set()

    labels, mapping = model_options()
    model_var = tk.StringVar(value=labels[0] if labels else "")
    input_var = tk.StringVar()
    output_var = tk.StringVar()
    gpu_var = tk.StringVar(value="0")

    body = ttk.Frame(dialog, padding=12)
    body.pack(fill=tk.BOTH, expand=True)
    ttk.Label(body, text="Model").grid(row=0, column=0, sticky="e", pady=4, padx=4)
    ttk.Combobox(body, textvariable=model_var, values=labels, state="readonly", width=52).grid(
        row=0, column=1, columnspan=3, sticky="w", pady=4
    )
    ttk.Label(body, text="Input").grid(row=1, column=0, sticky="e", pady=4, padx=4)
    ttk.Entry(body, textvariable=input_var, width=46).grid(row=1, column=1, sticky="w")
    ttk.Button(
        body,
        text="檔案",
        command=lambda: choose_file(input_var, [("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All", "*.*")]),
    ).grid(row=1, column=2, padx=3)
    ttk.Button(body, text="資料夾", command=lambda: choose_dir(input_var)).grid(
        row=1, column=3, padx=3
    )
    ttk.Label(body, text="Output").grid(row=2, column=0, sticky="e", pady=4, padx=4)
    ttk.Entry(body, textvariable=output_var, width=46).grid(row=2, column=1, sticky="w")
    ttk.Button(body, text="資料夾", command=lambda: choose_dir(output_var)).grid(
        row=2, column=2, padx=3
    )
    ttk.Label(body, text="gpu_id").grid(row=3, column=0, sticky="e", pady=4, padx=4)
    ttk.Entry(body, textvariable=gpu_var, width=10).grid(row=3, column=1, sticky="w")
    if not labels:
        ttk.Label(body, text="尚無模型，請先執行模型訓練").grid(row=4, column=1, sticky="w")

    def execute():
        model_id = mapping.get(model_var.get())
        if model_id is None:
            messagebox.showwarning("模型預測", "請先選擇 model", parent=dialog)
            return
        if not input_var.get().strip():
            messagebox.showwarning("模型預測", "請先選擇 input", parent=dialog)
            return

        def task():
            from predict_module import Predictor

            result = Predictor(
                model_id=model_id,
                input_path=input_var.get().strip(),
                output_dir=output_var.get().strip() or None,
                gpu_id=int(gpu_var.get()),
            ).run()
            return f"輸出位置：{result['output_dir']}"

        run_background(status_var, "模型預測完成", task)
        dialog.destroy()

    actions = ttk.Frame(dialog, padding=12)
    actions.pack(fill=tk.X)
    ttk.Button(actions, text="開始預測", command=execute).pack(side=tk.RIGHT)
    ttk.Button(actions, text="取消", command=dialog.destroy).pack(side=tk.RIGHT, padx=6)


def main():
    root = tk.Tk()
    root.title("Probe-Mark")
    root.geometry("540x300")
    root.resizable(False, False)

    ttk.Label(root, text="Probe-Mark", font=("Arial", 18, "bold")).pack(pady=(22, 6))
    ttk.Label(root, text="探針痕跡影像辨識").pack(pady=(0, 18))

    status_var = tk.StringVar(value="待命")

    btn_frame = ttk.Frame(root)
    btn_frame.pack(pady=4)
    ttk.Button(
        btn_frame,
        text="資料清洗",
        width=16,
        command=lambda: open_clean_dialog(root, status_var),
    ).grid(row=0, column=0, padx=8)
    ttk.Button(
        btn_frame,
        text="模型訓練",
        width=16,
        command=lambda: open_train_dialog(root, status_var),
    ).grid(row=0, column=1, padx=8)
    ttk.Button(
        btn_frame,
        text="模型預測",
        width=16,
        command=lambda: open_predict_dialog(root, status_var),
    ).grid(row=0, column=2, padx=8)

    ttk.Label(root, textvariable=status_var, foreground="blue").pack(pady=20)

    root.mainloop()


if __name__ == "__main__":
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    main()
