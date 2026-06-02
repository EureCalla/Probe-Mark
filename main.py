import logging
import json
import os
import re
import sys
import threading
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from zoneinfo import ZoneInfo

from mpivr20_cms import get_clean_output_dir, get_output_root_dir

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = get_output_root_dir()
OUTPUT_BASE = get_clean_output_dir()
EXCEL_EXTENSIONS = (".xlsx", ".xlsm")
EXCEL_FILETYPES = [("Excel files", "*.xlsx *.xlsm"), ("All files", "*.*")]

logger = logging.getLogger("probe_mark.gui")


def run_background(status_var, task_name, target, show_done_popup=True):
    done_label = f"{task_name}完成"
    fail_label = f"{task_name}失敗"

    def worker():
        try:
            status_var.set("執行中...")
            logger.info("%s 開始…", task_name)
            result = target()
            status_var.set(done_label)
            text = str(result) if result is not None else done_label
            logger.info("%s：%s", done_label, text)
            if show_done_popup:
                messagebox.showinfo(done_label, text)
        except Exception as exc:
            status_var.set(fail_label)
            logger.exception("%s：%s: %s", fail_label, type(exc).__name__, exc)
            messagebox.showerror(fail_label, f"{type(exc).__name__}: {exc}")

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


def safe_folder_name(name: str) -> str:
    """Return a Windows-safe folder name for user-entered run names."""
    return re.sub(r'[<>:"/\\\\|?*]+', "_", name.strip()) or "probe_mark"


def model_output_path(root_dir: str, run_name: str) -> str:
    """Build the model output folder path for the current Taiwan date."""
    date_text = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y%m%d")
    folder_name = f"run{{自動產生}}_{safe_folder_name(run_name)}_{date_text}"
    return os.path.join(root_dir, "model", folder_name)


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

        run_background(status_var, "資料清洗", task, show_done_popup=False)
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
    dialog.geometry("1240x820")
    dialog.minsize(1100, 720)
    dialog.transient(parent)
    dialog.grab_set()

    labels, mapping, sample_counts = dataset_options()
    model_root_var = tk.StringVar(value=OUTPUT_ROOT)
    model_path_var = tk.StringVar()
    sample_summary_var = tk.StringVar(
        value=(
            "排除: 0 個；Train/Test pool 有效: 0 / 無效: 0；"
            "Validation 有效: 0 / 無效: 0\n"
            "Validation 比例: 0.0%；Test 比例: 20.0%"
        )
    )
    field_defs = [
        ("run_name", "Run 名稱", "probe_mark", "本次訓練的名稱；會影響 runs/ 底下的輸出資料夾。"),
        ("split_name", "Split 名稱", "default", "資料切分名稱；相同 dataset + split 名稱會覆寫舊切分。"),
        ("test_ratio", "Test 比例", "0.2", "從 train/test pool 中切出多少比例做 final test；0.2 表示 80% train / 20% test。"),
        ("encoder_name", "Encoder", "resnet18", "特徵抽取 backbone，例如 resnet18、resnet50；需為 segmentation_models_pytorch 支援名稱。"),
        ("encoder_weights", "Encoder 權重", "imagenet", "Encoder 預訓練權重；常用 imagenet，留空表示不載入預訓練權重。"),
        ("decoder_name", "Decoder", "FPN", "分割模型架構，例如 FPN、Unet、DeepLabV3Plus。"),
        ("max_epochs", "Epochs", "10", "完整看過訓練資料的次數；越大訓練越久。"),
        ("batch_size", "Batch size", "16", "每次送進模型的影像張數；GPU 記憶體不足時調小。"),
        ("num_workers", "Workers", "0", "DataLoader 背景讀圖程序數；Windows/Tk 介面建議先用 0。"),
        ("lr", "Learning rate", "0.0001", "Adam optimizer 學習率；太大可能不穩，太小會學得慢。"),
        ("eta_min", "最低 LR", "0.00001", "CosineAnnealingLR 的最低 learning rate。"),
        ("gpu_id", "GPU ID", "-1", "使用哪張 GPU；-1 表示 CPU。"),
    ]
    fields = {key: tk.StringVar(value=default) for key, _label, default, _help in field_defs}

    def update_model_path(*_args) -> None:
        """Refresh the displayed model output folder."""
        run_name = fields["run_name"].get().strip() or "probe_mark"
        model_path_var.set(model_output_path(model_root_var.get().strip(), run_name))

    fields["run_name"].trace_add("write", update_model_path)
    model_root_var.trace_add("write", update_model_path)
    update_model_path()

    container = ttk.Frame(dialog)
    container.pack(fill=tk.BOTH, expand=True)

    body = ttk.Frame(container, padding=12)
    body.pack(side=tk.LEFT, fill=tk.Y, expand=False)
    body.columnconfigure(2, weight=1)

    plot_frame = ttk.LabelFrame(container, text="訓練即時曲線", padding=8)
    plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0, 12), pady=12)
    plot_status_var = tk.StringVar(value="尚未開始訓練")
    ttk.Label(plot_frame, textvariable=plot_status_var, foreground="#888").pack(anchor="w")
    plot_canvas_holder = ttk.Frame(plot_frame)
    plot_canvas_holder.pack(fill=tk.BOTH, expand=True)

    all_dataset_ids = [mapping[label] for label in labels]

    label_for = {dataset_id: label for label, dataset_id in mapping.items()}
    excluded_ids: set[int] = set()
    validation_selected_ids: set[int] = set()

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
    test_wrap, dataset_list = _build_dataset_listbox(dataset_frame, "Validation Dataset")
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
            if d in validation_selected_ids:
                dataset_list.selection_set(idx)
        excluded_count_var.set(f"已排除: {len(excluded_ids)}")
        update_sample_summary()

    def selected_validation_dataset_ids() -> list[int]:
        """Return dataset ids selected as validation datasets."""
        return [d for d in visible_dataset_ids() if d in validation_selected_ids]

    def selected_train_dataset_ids() -> list[int]:
        """Return visible dataset ids that are not selected as validation."""
        return [d for d in visible_dataset_ids() if d not in validation_selected_ids]

    def update_sample_summary(_event: tk.Event | None = None) -> None:
        """Refresh sample counts for excluded, train and test dataset groups."""
        train_ids = selected_train_dataset_ids()
        validation_ids = selected_validation_dataset_ids()
        train_pool_valid = sum(sample_counts[d]["active"] for d in train_ids)
        train_pool_invalid = sum(sample_counts[d]["failed"] for d in train_ids)
        validation_valid = sum(sample_counts[d]["active"] for d in validation_ids)
        validation_invalid = sum(sample_counts[d]["failed"] for d in validation_ids)
        total_visible_valid = train_pool_valid + validation_valid
        validation_ratio = validation_valid / total_visible_valid if total_visible_valid else 0.0
        try:
            test_ratio = float(fields["test_ratio"].get())
        except ValueError:
            test_ratio = 0.0
        sample_summary_var.set(
            f"排除: {len(excluded_ids)} 個；"
            f"Train/Test pool 有效: {train_pool_valid} / 無效: {train_pool_invalid}；"
            f"Validation 有效: {validation_valid} / 無效: {validation_invalid}\n"
            f"Validation 比例: {validation_ratio:.1%}；Test 比例: {test_ratio:.1%}"
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
        validation_selected_ids.difference_update(picks)
        refresh_listboxes()

    def on_restore() -> None:
        if not excluded_ids:
            return
        excluded_ids.clear()
        refresh_listboxes()

    def on_validation_selection_change(_event: tk.Event | None = None) -> None:
        visible = visible_dataset_ids()
        validation_selected_ids.clear()
        for i in dataset_list.curselection():
            validation_selected_ids.add(visible[i])
        update_sample_summary()

    exclude_btn.configure(command=on_exclude)
    restore_btn.configure(command=on_restore)
    fields["test_ratio"].trace_add("write", lambda *_args: update_sample_summary())
    dataset_list.bind("<<ListboxSelect>>", on_validation_selection_change)
    ttk.Button(
        body,
        text="?",
        width=3,
        command=lambda: messagebox.showinfo(
            "Dataset",
            "左側「可用資料集」：勾選後按【排除選取】，項目會從兩邊清單消失；\n"
            "按【復原全部】可把所有被排除的 dataset 拉回來。\n\n"
            "右側「Validation Dataset」：從可用資料集中勾選作為 validation。\n"
            "未排除且未選為 validation 的 dataset 會先作為 train/test pool，"
            "再依 Test 比例切成 train 與 final test。",
            parent=dialog,
        ),
    ).grid(row=0, column=1, sticky="nw", pady=4, padx=4)
    if labels:
        refresh_listboxes()
    else:
        sample_summary_var.set("尚無 dataset，請先執行資料清洗")

    ttk.Label(body, text="模型根目錄").grid(row=2, column=0, sticky="e", pady=3, padx=4)
    ttk.Entry(body, textvariable=model_root_var, width=48).grid(
        row=2, column=2, sticky="w", pady=3
    )
    ttk.Button(body, text="資料夾", command=lambda: choose_dir(model_root_var)).grid(
        row=2, column=1, sticky="w", pady=3, padx=4
    )
    ttk.Label(body, text="儲存位置").grid(row=3, column=0, sticky="e", pady=3, padx=4)
    ttk.Label(body, textvariable=model_path_var, foreground="#1e6fba", wraplength=520).grid(
        row=3, column=2, sticky="w", pady=3
    )

    for row, (key, label, _default, help_text) in enumerate(field_defs, start=4):
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

    # ─── matplotlib 圖表 ─────────────────────────────────────────────
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    fig = Figure(figsize=(5.2, 5.5), dpi=90)
    ax_loss = fig.add_subplot(2, 1, 1)
    ax_iou = fig.add_subplot(2, 1, 2)

    def _init_plot_axes() -> None:
        ax_loss.clear()
        ax_loss.set_title("Loss")
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("loss")
        ax_loss.grid(True, alpha=0.3)
        ax_iou.clear()
        ax_iou.set_title("IoU")
        ax_iou.set_xlabel("epoch")
        ax_iou.set_ylabel("iou")
        ax_iou.grid(True, alpha=0.3)
        fig.tight_layout()

    _init_plot_axes()
    plot_canvas = FigureCanvasTkAgg(fig, master=plot_canvas_holder)
    plot_canvas.draw()
    plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def redraw_plot(epoch_rows: list[dict]) -> None:
        if not epoch_rows:
            return
        epochs = [r["epoch"] for r in epoch_rows]
        train_loss = [r["train_loss"] for r in epoch_rows]
        val_loss = [r["val_loss"] for r in epoch_rows]
        train_iou = [r["train_iou"] for r in epoch_rows]
        val_iou = [r["val_iou"] for r in epoch_rows]
        _init_plot_axes()
        ax_loss.plot(epochs, train_loss, "-o", color="#1e6fba", label="train")
        ax_loss.plot(epochs, val_loss, "-o", color="#c0392b", label="val")
        ax_loss.legend(loc="best")
        ax_iou.plot(epochs, train_iou, "-o", color="#1e6fba", label="train")
        ax_iou.plot(epochs, val_iou, "-o", color="#27ae60", label="val")
        ax_iou.legend(loc="best")
        plot_canvas.draw_idle()

    # ─── 控制狀態（thread + run_id 追蹤） ────────────────────────────
    state = {
        "thread": None,
        "result": None,
        "error": None,
        "prev_max_run_id": None,
        "current_run_id": None,
    }

    def _set_inputs_state(state_value: str) -> None:
        """Toggle all input widgets in body recursively (Entry / Listbox / Button / Combobox)."""
        def walk(widget):
            for child in widget.winfo_children():
                cls = child.__class__.__name__
                if cls in ("Entry", "TEntry", "Combobox", "TCombobox", "Listbox",
                           "Button", "TButton", "Checkbutton", "TCheckbutton"):
                    try:
                        child.configure(state=state_value)
                    except tk.TclError:
                        pass
                walk(child)
        walk(body)

    def execute():
        train_dataset_ids = selected_train_dataset_ids()
        validation_dataset_ids = selected_validation_dataset_ids()
        if not validation_dataset_ids:
            messagebox.showwarning(
                "模型訓練", "請先選擇至少一個 validation dataset", parent=dialog
            )
            return
        if not train_dataset_ids:
            messagebox.showwarning("模型訓練", "至少需要保留一個 train dataset", parent=dialog)
            return
        try:
            test_ratio = float(fields["test_ratio"].get())
        except ValueError:
            messagebox.showwarning("模型訓練", "Test 比例必須是數字", parent=dialog)
            return
        if not 0 < test_ratio < 1:
            messagebox.showwarning("模型訓練", "Test 比例需介於 0 和 1 之間", parent=dialog)
            return
        model_output_root = model_root_var.get().strip()
        if not model_output_root:
            messagebox.showwarning("模型訓練", "模型根目錄不可空白", parent=dialog)
            return

        from database import DBManager
        db = DBManager()
        db.init_db()
        state["prev_max_run_id"] = db.get_latest_training_run_id() or 0
        state["current_run_id"] = None
        state["result"] = None
        state["error"] = None

        def worker():
            try:
                from train_module import Trainer
                logger.info("模型訓練 開始…")
                state["result"] = Trainer(
                    dataset_ids=train_dataset_ids,
                    validation_dataset_ids=validation_dataset_ids,
                    split_name=fields["split_name"].get().strip() or "default",
                    run_name=fields["run_name"].get().strip() or None,
                    test_ratio=test_ratio,
                    encoder_name=fields["encoder_name"].get().strip(),
                    encoder_weights=fields["encoder_weights"].get().strip() or None,
                    decoder_name=fields["decoder_name"].get().strip(),
                    max_epochs=int(fields["max_epochs"].get()),
                    batch_size=int(fields["batch_size"].get()),
                    num_workers=int(fields["num_workers"].get()),
                    lr=float(fields["lr"].get()),
                    eta_min=float(fields["eta_min"].get()),
                    gpu_id=int(fields["gpu_id"].get()),
                    model_output_root=model_output_root,
                ).run()
            except Exception as exc:
                state["error"] = exc

        _set_inputs_state("disabled")
        start_btn.configure(state="disabled")
        close_btn.configure(text="關閉（訓練中無法關閉）", state="disabled")
        plot_status_var.set("訓練啟動中…")
        status_var.set("執行中...")

        t = threading.Thread(target=worker, daemon=True)
        state["thread"] = t
        t.start()
        dialog.after(1000, _poll_training)

    def _poll_training() -> None:
        # 找出本次 run_id
        if state["current_run_id"] is None:
            try:
                from database import DBManager
                db = DBManager()
                with db._connect() as conn:
                    row = conn.execute(
                        "SELECT id FROM training_runs WHERE id > ? ORDER BY id ASC LIMIT 1",
                        (state["prev_max_run_id"],),
                    ).fetchone()
                if row:
                    state["current_run_id"] = row[0]
                    plot_status_var.set(f"訓練中（run #{row[0]}）— 每秒更新")
            except Exception as exc:
                logger.warning("輪詢 training_runs 失敗：%s", exc)

        # 取 epoch 指標
        if state["current_run_id"] is not None:
            try:
                from database import DBManager
                db = DBManager()
                rows = db.list_epoch_metrics(state["current_run_id"])
                if rows:
                    redraw_plot(rows)
                    plot_status_var.set(
                        f"訓練中（run #{state['current_run_id']}）— 已完成 {len(rows)} epochs"
                    )
            except Exception as exc:
                logger.warning("讀 epoch metrics 失敗：%s", exc)

        if state["thread"] is not None and state["thread"].is_alive():
            dialog.after(1000, _poll_training)
        else:
            _finalize_training()

    def _finalize_training() -> None:
        _set_inputs_state("normal")
        start_btn.configure(state="normal")
        close_btn.configure(text="關閉", state="normal")
        if state["error"] is not None:
            exc = state["error"]
            logger.exception("模型訓練失敗：%s: %s", type(exc).__name__, exc)
            messagebox.showerror("模型訓練失敗", f"{type(exc).__name__}: {exc}", parent=dialog)
            status_var.set("模型訓練失敗")
            plot_status_var.set("訓練失敗")
        else:
            result = state["result"]
            duration_text = None
            if isinstance(result, dict):
                run_id_done = result.get("run_id", state["current_run_id"])
                model_id_done = result.get("model_id")
                duration = result.get("duration_seconds")
                if duration is not None:
                    hh, rem = divmod(int(duration), 3600)
                    mm, ss = divmod(rem, 60)
                    duration_text = f"{hh}:{mm:02d}:{ss:02d}"
                text = (
                    f"run_id={run_id_done}, model_id={model_id_done}, "
                    f"耗時 {duration_text or '—'}"
                )
            else:
                text = str(result) if result is not None else "完成"
            logger.info("模型訓練完成：%s", text)
            messagebox.showinfo("模型訓練完成", text, parent=dialog)
            status_var.set("模型訓練完成")
            run_id = state["current_run_id"]
            if duration_text:
                plot_status_var.set(
                    f"訓練完成（run #{run_id}，耗時 {duration_text}）"
                    if run_id
                    else f"訓練完成（耗時 {duration_text}）"
                )
            else:
                plot_status_var.set(
                    f"訓練完成（run #{run_id}）" if run_id else "訓練完成"
                )

    def on_close():
        if state["thread"] is not None and state["thread"].is_alive():
            messagebox.showinfo(
                "模型訓練",
                "訓練進行中，請等待完成（暫停功能將於後續版本提供）",
                parent=dialog,
            )
            return
        dialog.destroy()

    actions = ttk.Frame(dialog, padding=12)
    actions.pack(fill=tk.X)
    start_btn = ttk.Button(actions, text="開始訓練", command=execute)
    start_btn.pack(side=tk.RIGHT)
    close_btn = ttk.Button(actions, text="關閉", command=on_close)
    close_btn.pack(side=tk.RIGHT, padx=6)
    dialog.protocol("WM_DELETE_WINDOW", on_close)


def open_predict_dialog(parent, status_var):
    dialog = tk.Toplevel(parent)
    dialog.title("模型預測")
    dialog.geometry("980x760")
    dialog.minsize(860, 640)
    dialog.transient(parent)
    dialog.grab_set()

    labels, mapping = model_options()
    model_var = tk.StringVar(value=labels[0] if labels else "")
    input_var = tk.StringVar()
    output_var = tk.StringVar()
    gpu_var = tk.StringVar(value="-1")

    body = ttk.Frame(dialog, padding=12)
    body.pack(fill=tk.BOTH, expand=True)
    body.columnconfigure(1, weight=1)
    body.rowconfigure(4, weight=1)
    ttk.Label(body, text="Model").grid(row=0, column=0, sticky="e", pady=4, padx=4)
    model_combo = ttk.Combobox(body, textvariable=model_var, values=labels, state="readonly", width=52)
    model_combo.grid(
        row=0, column=1, columnspan=3, sticky="ew", pady=4
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
        ttk.Label(body, text="尚無模型，請先執行模型訓練").grid(row=5, column=1, sticky="w")

    metrics_frame = ttk.LabelFrame(body, text="Training Metrics", padding=8)
    metrics_frame.grid(row=4, column=0, columnspan=4, sticky="nsew", pady=(10, 0))
    metrics_frame.rowconfigure(0, weight=1)
    metrics_frame.columnconfigure(0, weight=1)
    metric_status_var = tk.StringVar(value="")
    ttk.Label(
        metrics_frame,
        textvariable=metric_status_var,
        foreground="#444",
        justify=tk.LEFT,
        wraplength=900,
    ).grid(
        row=1, column=0, sticky="w", pady=(6, 0)
    )

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    metric_fig = Figure(figsize=(8.4, 4.0), dpi=90)
    metric_ax_loss = metric_fig.add_subplot(1, 2, 1)
    metric_ax_iou = metric_fig.add_subplot(1, 2, 2)

    def init_metric_axes(message=None):
        metric_ax_loss.clear()
        metric_ax_loss.set_title("Loss / epoch")
        metric_ax_loss.set_xlabel("epoch")
        metric_ax_loss.set_ylabel("loss")
        metric_ax_loss.grid(True, alpha=0.3)
        metric_ax_iou.clear()
        metric_ax_iou.set_title("IoU & Dice / epoch")
        metric_ax_iou.set_xlabel("epoch")
        metric_ax_iou.set_ylabel("score")
        metric_ax_iou.grid(True, alpha=0.3)
        if message:
            metric_ax_loss.text(0.5, 0.5, message, transform=metric_ax_loss.transAxes, ha="center", va="center")
            metric_ax_iou.text(0.5, 0.5, message, transform=metric_ax_iou.transAxes, ha="center", va="center")
        metric_fig.tight_layout()

    init_metric_axes("Select Model")
    metric_canvas = FigureCanvasTkAgg(metric_fig, master=metrics_frame)
    metric_canvas.draw()
    metric_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def refresh_model_metrics(*_):
        model_id = mapping.get(model_var.get())
        if model_id is None:
            init_metric_axes("No model")
            metric_status_var.set("")
            metric_canvas.draw_idle()
            return
        from database import DBManager

        db = DBManager()
        db.init_db()
        model = db.get_model(model_id)
        if not model:
            init_metric_axes("Model not found")
            metric_status_var.set("")
            metric_canvas.draw_idle()
            return
        epoch_rows = db.list_epoch_metrics(model["run_id"])
        if not epoch_rows:
            init_metric_axes("No epoch metrics")
            metric_status_var.set(f"model#{model_id} / run#{model['run_id']} 沒有 epoch metrics 紀錄")
            metric_canvas.draw_idle()
            return

        epochs = [row["epoch"] for row in epoch_rows]
        init_metric_axes()
        metric_ax_loss.plot(epochs, [row["train_loss"] for row in epoch_rows], "-o", label="train")
        metric_ax_loss.plot(epochs, [row["val_loss"] for row in epoch_rows], "-o", label="validation")
        metric_ax_loss.legend(loc="best")
        metric_ax_iou.plot(epochs, [row["train_iou"] for row in epoch_rows], "-o", label="train")
        metric_ax_iou.plot(epochs, [row["val_iou"] for row in epoch_rows], "-o", label="validation")
        metric_ax_iou.legend(loc="best")

        val_ious = [row["val_iou"] for row in epoch_rows if row["val_iou"] is not None]
        best_val_iou = max(val_ious) if val_ious else None
        suffix = f"，best validation IoU={best_val_iou:.4f}" if best_val_iou is not None else ""
        metric_status_var.set(f"model#{model_id} / run#{model['run_id']}，epochs={len(epoch_rows)}{suffix}")
        metric_canvas.draw_idle()

    model_combo.bind("<<ComboboxSelected>>", refresh_model_metrics)
    refresh_model_metrics()

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

        run_background(status_var, "模型預測", task)
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
