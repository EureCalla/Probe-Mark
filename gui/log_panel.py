"""GUI log panel: thread-safe ScrolledText sink for logging + stdout."""

from __future__ import annotations

import datetime as _dt
import logging
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

_LEVEL_TAGS = {
    "DEBUG": ("DEBUG", "#888888"),
    "INFO": ("INFO", "#1e6fba"),
    "WARNING": ("WARN", "#c97b00"),
    "WARN": ("WARN", "#c97b00"),
    "ERROR": ("ERROR", "#c0392b"),
    "CRITICAL": ("ERROR", "#c0392b"),
}


class LogPanel(ttk.Frame):
    """A log viewer panel with colored level tags and clear/save toolbar."""

    def __init__(self, master, *, font=("Consolas", 10), **kwargs):
        super().__init__(master, **kwargs)

        bar = ttk.Frame(self)
        bar.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(bar, text="執行紀錄", font=("Microsoft JhengHei", 10, "bold")).pack(
            side=tk.LEFT
        )
        ttk.Button(bar, text="存檔", width=8, command=self.save_as).pack(side=tk.RIGHT)
        ttk.Button(bar, text="清空", width=8, command=self.clear).pack(
            side=tk.RIGHT, padx=(0, 6)
        )

        self.text = ScrolledText(
            self,
            wrap=tk.WORD,
            font=font,
            state=tk.DISABLED,
            background="#1e1e1e",
            foreground="#dddddd",
            insertbackground="#dddddd",
        )
        self.text.pack(fill=tk.BOTH, expand=True)

        for _label, color in _LEVEL_TAGS.values():
            self.text.tag_config(color, foreground=color)
        self.text.tag_config("ts", foreground="#7a7a7a")

    def append(self, level: str, message: str) -> None:
        """Thread-safe: schedule the write on the Tk main loop."""
        try:
            self.after(0, self._write, level, message)
        except RuntimeError:
            # Tk root already torn down; fall back to stderr-original.
            pass

    def _write(self, level: str, message: str) -> None:
        label, color = _LEVEL_TAGS.get(level.upper(), ("INFO", "#1e6fba"))
        ts = _dt.datetime.now().strftime("%H:%M:%S")
        self.text.configure(state=tk.NORMAL)
        self.text.insert(tk.END, f"{ts}  ", ("ts",))
        self.text.insert(tk.END, f"[{label:<5}] ", (color,))
        self.text.insert(tk.END, f"{message}\n")
        self.text.see(tk.END)
        self.text.configure(state=tk.DISABLED)

    def clear(self) -> None:
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.configure(state=tk.DISABLED)

    def save_as(self) -> None:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            defaultextension=".log",
            initialfile=f"probe_mark_{ts}.log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        content = self.text.get("1.0", tk.END)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as exc:
            messagebox.showerror("存檔失敗", str(exc), parent=self)
            return
        messagebox.showinfo("存檔完成", f"已寫入：{path}", parent=self)


class TkLogHandler(logging.Handler):
    """Route logging records into a LogPanel."""

    def __init__(self, panel: LogPanel):
        super().__init__()
        self.panel = panel
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        self.panel.append(record.levelname, msg)


class StreamToLogger:
    """File-like wrapper that pipes write() into a logger at a fixed level."""

    def __init__(self, logger: logging.Logger, level: int = logging.INFO):
        self.logger = logger
        self.level = level
        self._buf = ""

    def write(self, data) -> int:
        if not isinstance(data, str):
            try:
                data = data.decode("utf-8", errors="replace")
            except Exception:
                data = str(data)
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip()
            if line:
                self.logger.log(self.level, line)
        return len(data)

    def flush(self) -> None:
        if self._buf.strip():
            self.logger.log(self.level, self._buf.rstrip())
        self._buf = ""

    def isatty(self) -> bool:
        return False


def install_log_panel(
    master,
    *,
    logger_name: str = "probe_mark",
    redirect_stdio: bool = True,
) -> LogPanel:
    """Create a LogPanel, attach it to the named logger, and (optionally) hijack stdio."""
    panel = LogPanel(master)

    handler = TkLogHandler(panel)
    handler.setLevel(logging.DEBUG)

    logging.getLogger(logger_name).setLevel(logging.DEBUG)

    root_logger = logging.getLogger()
    if root_logger.level in (logging.WARNING, 0):
        root_logger.setLevel(logging.INFO)
    if not any(isinstance(h, TkLogHandler) for h in root_logger.handlers):
        root_logger.addHandler(handler)

    if redirect_stdio:
        panel._orig_stdout = sys.stdout
        panel._orig_stderr = sys.stderr
        sys.stdout = StreamToLogger(logging.getLogger(f"{logger_name}.stdout"), logging.INFO)
        sys.stderr = StreamToLogger(logging.getLogger(f"{logger_name}.stderr"), logging.ERROR)

    return panel
