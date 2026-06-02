import logging
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from mpivr20_cms import get_clean_output_dir, get_output_root_dir

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = get_output_root_dir()
OUTPUT_BASE = get_clean_output_dir()
EXCEL_EXTENSIONS = (".xlsx", ".xlsm")
EXCEL_FILETYPES = [("Excel files", "*.xlsx *.xlsm"), ("All files", "*.*")]

logger = logging.getLogger("probe_mark.gui")


def run_background(status_var, done_message, target, show_done_popup=True):
    def worker():
        try:
            status_var.set("執行中...")
            logger.info("%s 開始…", done_message)
            result = target()
            status_var.set(done_message)
            text = str(result) if result is not None else done_message
            logger.info("%s：%s", done_message, text)
            if show_done_popup:
                messagebox.showinfo(done_message, text)
        except Exception as exc:
            status_var.set("執行失敗")
            logger.exception("%s 失敗：%s: %s", done_message, type(exc).__name__, exc)
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
    """Build the final clean log summary."""
    lines = [
        f"成功 dataset ids: {result['dataset_ids']}",
        f"成功樣本數: {result['total_samples']}",
        f"略過檔案: {len(result.get('failed', []))}",
    ]
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
    max_area_percent_var = tk.StringVar(value="50")
    min_pixels_var = tk.StringVar(value="30")

    top = ttk.Frame(dialog, padding=12)
    top.pack(fill=tk.X)
    ttk.Label(top, text=f"輸出根目錄：{OUTPUT_ROOT}（清洗資料：data）").pack(side=tk.LEFT)
    ttk.Checkbutton(top, text="強制重建快取", variable=force_var).pack(side=tk.RIGHT)

    advanced = ttk.LabelFrame(dialog, text="進階清洗規則", padding=8)
    advanced.pack(fill=tk.X, padx=12, pady=(0, 8))
    ttk.Label(advanced, text="ground_truth 最大面積 (%)").grid(
        row=0, column=0, sticky="w", padx=4
    )
    ttk.Entry(advanced, textvariable=max_area_percent_var, width=8).grid(
        row=0, column=1, sticky="w", padx=4
    )
    ttk.Label(advanced, text="ground_truth 最小 pixels").grid(
        row=0, column=2, sticky="w", padx=12
    )
    ttk.Entry(advanced, textvariable=min_pixels_var, width=8).grid(
        row=0, column=3, sticky="w", padx=4
    )

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
        try:
            max_ground_truth_ratio = float(max_area_percent_var.get()) / 100.0
            min_ground_truth_pixels = int(min_pixels_var.get())
        except ValueError:
            messagebox.showwarning("資料清洗", "進階清洗規則必須是數字", parent=dialog)
            return
        if not 0 < max_ground_truth_ratio <= 1:
            messagebox.showwarning(
                "資料清洗", "ground_truth 最大面積需介於 1 到 100", parent=dialog
            )
            return
        if min_ground_truth_pixels < 0:
            messagebox.showwarning(
                "資料清洗", "ground_truth 最小 pixels 不可小於 0", parent=dialog
            )
            return

        def task():
            from load_clean_module import LoadCleanService

            service = LoadCleanService(
                tasks=tasks,
                save_dir=OUTPUT_BASE,
                force=force_var.get(),
                max_ground_truth_ratio=max_ground_truth_ratio,
                min_ground_truth_pixels=min_ground_truth_pixels,
            )
            result = service.run()
            return format_clean_result(result)

        run_background(status_var, "資料清洗完成", task, show_done_popup=False)
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
    counts = {}
    for dataset in datasets:
        sample_counts = db.count_samples_by_status(dataset["id"])
        label = (
            f"#{dataset['id']} {dataset['dataset_name']} "
            f"({sample_counts['active']} valid / {sample_counts['failed']} invalid)"
        )
        labels.append(label)
        mapping[label] = dataset["id"]
        counts[dataset["id"]] = sample_counts
    return labels, mapping, counts


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
    dialog.geometry("880x600")
    dialog.transient(parent)
    dialog.grab_set()

    labels, mapping, sample_counts = dataset_options()
    sample_summary_var = tk.StringVar(
        value="排除: 0 個；Train 有效: 0 / 無效: 0；Test 有效: 0 / 無效: 0\n驗證比例: 0.0%"
    )
    field_defs = [
        ("run_name", "Run 名稱", "probe_mark", "本次訓練的名稱；會影響 runs/ 底下的輸出資料夾。"),
        ("split_name", "Split 名稱", "default", "資料切分名稱；相同 dataset + split 名稱會覆寫舊切分。"),
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
    all_dataset_ids = [mapping[label] for label in labels]

    label_for = {dataset_id: label for label, dataset_id in mapping.items()}
    excluded_ids: set[int] = set()
    test_selected_ids: set[int] = set()

    ttk.Label(body, text="資料集選擇").grid(row=0, column=0, sticky="ne", pady=4, padx=4)
    dataset_frame = ttk.Frame(body)
    dataset_frame.grid(row=0, column=2, sticky="nsew", pady=4, padx=4)
    dataset_frame.columnconfigure(0, weight=1)
    dataset_frame.columnconfigure(1, weight=1)
    dataset_frame.rowconfigure(0, weight=1)

    def _build_dataset_listbox(parent, title: str) -> tuple[ttk.LabelFrame, tk.Listbox]:
        wrap = ttk.LabelFrame(parent, text=title, padding=4)
        listbox = tk.Listbox(
            wrap,
            selectmode=tk.EXTENDED,
            height=7,
            exportselection=False,
        )
        sb = ttk.Scrollbar(wrap, orient=tk.VERTICAL, command=listbox.yview)
        listbox.configure(yscrollcommand=sb.set)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        return wrap, listbox

    candidate_wrap, candidate_list = _build_dataset_listbox(dataset_frame, "可用資料集")
    candidate_wrap.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
    test_wrap, dataset_list = _build_dataset_listbox(dataset_frame, "Test Dataset")
    test_wrap.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

    candidate_actions = ttk.Frame(dataset_frame)
    candidate_actions.grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(4, 0))
    exclude_btn = ttk.Button(candidate_actions, text="排除選取")
    exclude_btn.pack(side=tk.LEFT)
    restore_btn = ttk.Button(candidate_actions, text="復原全部")
    restore_btn.pack(side=tk.LEFT, padx=6)
    excluded_count_var = tk.StringVar(value="已排除: 0")
    ttk.Label(candidate_actions, textvariable=excluded_count_var, foreground="#888").pack(
        side=tk.RIGHT
    )

    ttk.Label(body, textvariable=sample_summary_var, foreground="#1e6fba").grid(
        row=1, column=2, sticky="w", padx=4, pady=(4, 6)
    )

    def visible_dataset_ids() -> list[int]:
        return [d for d in all_dataset_ids if d not in excluded_ids]

    def refresh_listboxes() -> None:
        visible = visible_dataset_ids()
        candidate_list.delete(0, tk.END)
        for d in visible:
            candidate_list.insert(tk.END, label_for[d])
        dataset_list.delete(0, tk.END)
        for idx, d in enumerate(visible):
            dataset_list.insert(tk.END, label_for[d])
            if d in test_selected_ids:
                dataset_list.selection_set(idx)
        excluded_count_var.set(f"已排除: {len(excluded_ids)}")
        update_sample_summary()

    def selected_test_dataset_ids() -> list[int]:
        """Return dataset ids selected as test (from the canonical set)."""
        return [d for d in visible_dataset_ids() if d in test_selected_ids]

    def selected_train_dataset_ids() -> list[int]:
        """Return visible dataset ids that are not selected as test."""
        return [d for d in visible_dataset_ids() if d not in test_selected_ids]

    def update_sample_summary(_event: tk.Event | None = None) -> None:
        """Refresh sample counts for excluded, train and test dataset groups."""
        train_ids = selected_train_dataset_ids()
        test_ids = selected_test_dataset_ids()
        train_valid = sum(sample_counts[d]["active"] for d in train_ids)
        train_invalid = sum(sample_counts[d]["failed"] for d in train_ids)
        test_valid = sum(sample_counts[d]["active"] for d in test_ids)
        test_invalid = sum(sample_counts[d]["failed"] for d in test_ids)
        total_valid = train_valid + test_valid
        validation_ratio = test_valid / total_valid if total_valid else 0.0
        sample_summary_var.set(
            f"排除: {len(excluded_ids)} 個；"
            f"Train 有效: {train_valid} / 無效: {train_invalid}；"
            f"Test 有效: {test_valid} / 無效: {test_invalid}\n"
            f"驗證比例: {validation_ratio:.1%}"
        )

    def on_exclude() -> None:
        visible = visible_dataset_ids()
        picks = [visible[i] for i in candidate_list.curselection()]
        if not picks:
            messagebox.showwarning(
                "排除", "請先在「可用資料集」中勾選要排除的項目", parent=dialog
            )
            return
        excluded_ids.update(picks)
        test_selected_ids.difference_update(picks)
        refresh_listboxes()

    def on_restore() -> None:
        if not excluded_ids:
            return
        excluded_ids.clear()
        refresh_listboxes()

    def on_test_selection_change(_event: tk.Event | None = None) -> None:
        visible = visible_dataset_ids()
        test_selected_ids.clear()
        for i in dataset_list.curselection():
            test_selected_ids.add(visible[i])
        update_sample_summary()

    exclude_btn.configure(command=on_exclude)
    restore_btn.configure(command=on_restore)
    dataset_list.bind("<<ListboxSelect>>", on_test_selection_change)
    ttk.Button(
        body,
        text="?",
        width=3,
        command=lambda: messagebox.showinfo(
            "Dataset",
            "左側「可用資料集」：勾選後按【排除選取】，項目會從兩邊清單消失；\n"
            "按【復原全部】可把所有被排除的 dataset 拉回來。\n\n"
            "右側「Test Dataset」：從可用資料集中勾選作為 test；其餘為 train。",
            parent=dialog,
        ),
    ).grid(row=0, column=1, sticky="nw", pady=4, padx=4)
    if labels:
        refresh_listboxes()
    else:
        sample_summary_var.set("尚無 dataset，請先執行資料清洗")

    for row, (key, label, _default, help_text) in enumerate(field_defs, start=2):
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
        train_dataset_ids = selected_train_dataset_ids()
        test_dataset_ids = selected_test_dataset_ids()
        if not test_dataset_ids:
            messagebox.showwarning("模型訓練", "請先選擇至少一個 test dataset", parent=dialog)
            return
        if not train_dataset_ids:
            messagebox.showwarning("模型訓練", "至少需要保留一個 train dataset", parent=dialog)
            return

        def task():
            from train_module import Trainer

            return Trainer(
                dataset_ids=train_dataset_ids,
                test_dataset_ids=test_dataset_ids,
                split_name=fields["split_name"].get().strip() or "default",
                run_name=fields["run_name"].get().strip() or None,
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
    from gui.log_panel import install_log_panel

    root = tk.Tk()
    root.title("Probe-Mark")
    root.geometry("860x600")
    root.minsize(720, 480)

    header = ttk.Frame(root, padding=(14, 10, 14, 4))
    header.pack(fill=tk.X)
    ttk.Label(header, text="Probe-Mark", font=("Arial", 16, "bold")).pack(side=tk.LEFT)
    ttk.Label(header, text="  探針痕跡影像辨識", foreground="#666").pack(
        side=tk.LEFT, padx=(4, 0)
    )

    status_var = tk.StringVar(value="待命")

    ctrl = ttk.Frame(root, padding=(14, 4, 14, 8))
    ctrl.pack(fill=tk.X)
    ttk.Button(
        ctrl,
        text="資料清洗",
        width=14,
        command=lambda: open_clean_dialog(root, status_var),
    ).pack(side=tk.LEFT, padx=(0, 6))
    ttk.Button(
        ctrl,
        text="模型訓練",
        width=14,
        command=lambda: open_train_dialog(root, status_var),
    ).pack(side=tk.LEFT, padx=6)
    ttk.Button(
        ctrl,
        text="模型預測",
        width=14,
        command=lambda: open_predict_dialog(root, status_var),
    ).pack(side=tk.LEFT, padx=6)
    ttk.Label(ctrl, text="狀態：", foreground="#666").pack(side=tk.LEFT, padx=(16, 0))
    ttk.Label(ctrl, textvariable=status_var, foreground="#1e6fba").pack(side=tk.LEFT)

    ttk.Separator(root, orient=tk.HORIZONTAL).pack(fill=tk.X)

    log_panel = install_log_panel(root, logger_name="probe_mark")
    log_panel.pack(fill=tk.BOTH, expand=True, padx=14, pady=(8, 12))

    logger.info("Probe-Mark 啟動完成，等待操作")
    root.mainloop()


if __name__ == "__main__":
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    main()
