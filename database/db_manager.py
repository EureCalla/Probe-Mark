import json
import os
import sqlite3
from pathlib import Path

from mpivr20_cms import get_database_path


class DBManager:
    DB_PATH = get_database_path()

    def __init__(self, db_path=None):
        self.db_path = str(db_path or self.DB_PATH)

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init_db(self):
        if not self.db_path.startswith("\\\\"):
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS source_excels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name TEXT NOT NULL,
                    original_path TEXT NOT NULL,
                    output_name TEXT NOT NULL,
                    file_mtime REAL,
                    file_size INTEGER,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT DEFAULT (datetime('now','localtime')),
                    updated_at TEXT DEFAULT (datetime('now','localtime')),
                    UNIQUE(original_path, output_name)
                );

                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_name TEXT NOT NULL,
                    source_excel_id INTEGER NOT NULL,
                    processed_dir TEXT NOT NULL,
                    storage_layout TEXT NOT NULL DEFAULT 'legacy_nested',
                    n_samples INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'running',
                    created_at TEXT DEFAULT (datetime('now','localtime')),
                    updated_at TEXT DEFAULT (datetime('now','localtime')),
                    UNIQUE(source_excel_id, dataset_name),
                    FOREIGN KEY (source_excel_id) REFERENCES source_excels(id)
                );

                CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id INTEGER NOT NULL,
                    sample_name TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    ground_truth_path TEXT NOT NULL,
                    label_path TEXT,
                    mask_view_path TEXT,
                    width INTEGER,
                    height INTEGER,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT DEFAULT (datetime('now','localtime')),
                    UNIQUE(dataset_id, sample_name),
                    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS dataset_splits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id INTEGER NOT NULL,
                    split_name TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    train_ratio REAL NOT NULL,
                    val_ratio REAL NOT NULL,
                    test_ratio REAL NOT NULL,
                    created_at TEXT DEFAULT (datetime('now','localtime')),
                    UNIQUE(dataset_id, split_name),
                    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS sample_splits (
                    split_id INTEGER NOT NULL,
                    sample_id INTEGER NOT NULL,
                    split TEXT NOT NULL,
                    PRIMARY KEY (split_id, sample_id),
                    FOREIGN KEY (split_id) REFERENCES dataset_splits(id) ON DELETE CASCADE,
                    FOREIGN KEY (sample_id) REFERENCES samples(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_name TEXT,
                    dataset_id INTEGER NOT NULL,
                    split_id INTEGER NOT NULL,
                    encoder_name TEXT NOT NULL,
                    decoder_name TEXT NOT NULL,
                    epochs INTEGER,
                    batch_size INTEGER,
                    lr REAL,
                    status TEXT NOT NULL DEFAULT 'running',
                    val_metric REAL,
                    duration_seconds REAL,
                    created_at TEXT DEFAULT (datetime('now','localtime')),
                    updated_at TEXT DEFAULT (datetime('now','localtime')),
                    FOREIGN KEY (dataset_id) REFERENCES datasets(id),
                    FOREIGN KEY (split_id) REFERENCES dataset_splits(id)
                );

                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    model_dir TEXT NOT NULL,
                    best_model_path TEXT,
                    snapshot_path TEXT,
                    opt_path TEXT,
                    parameter_count INTEGER,
                    db_import_test_mean_iou REAL,
                    db_import_test_n_samples INTEGER,
                    db_import_test_prediction_run_id INTEGER,
                    db_import_test_updated_at TEXT,
                    created_at TEXT DEFAULT (datetime('now','localtime')),
                    FOREIGN KEY (run_id) REFERENCES training_runs(id)
                );

                CREATE TABLE IF NOT EXISTS training_run_params (
                    run_id INTEGER NOT NULL,
                    param_name TEXT NOT NULL,
                    param_value TEXT,
                    PRIMARY KEY (run_id, param_name),
                    FOREIGN KEY (run_id) REFERENCES training_runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS training_epoch_metrics (
                    run_id INTEGER NOT NULL,
                    epoch INTEGER NOT NULL,
                    train_loss REAL,
                    train_ap REAL,
                    train_iou REAL,
                    train_dice REAL,
                    val_loss REAL,
                    val_ap REAL,
                    val_iou REAL,
                    val_dice REAL,
                    lr REAL,
                    created_at TEXT DEFAULT (datetime('now','localtime')),
                    PRIMARY KEY (run_id, epoch),
                    FOREIGN KEY (run_id) REFERENCES training_runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS training_test_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    split_id INTEGER NOT NULL,
                    n_samples INTEGER NOT NULL,
                    test_loss REAL,
                    test_iou REAL,
                    test_dice REAL,
                    test_ap REAL,
                    created_at TEXT DEFAULT (datetime('now','localtime')),
                    FOREIGN KEY (run_id) REFERENCES training_runs(id) ON DELETE CASCADE,
                    FOREIGN KEY (split_id) REFERENCES dataset_splits(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS prediction_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER NOT NULL,
                    input_path TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'running',
                    created_at TEXT DEFAULT (datetime('now','localtime')),
                    updated_at TEXT DEFAULT (datetime('now','localtime')),
                    FOREIGN KEY (model_id) REFERENCES models(id)
                );

                CREATE TABLE IF NOT EXISTS prediction_outputs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_run_id INTEGER NOT NULL,
                    image_path TEXT NOT NULL,
                    mask_path TEXT NOT NULL,
                    overlay_path TEXT,
                    compare_path TEXT NOT NULL,
                    iou REAL,
                    status TEXT NOT NULL DEFAULT 'done',
                    error_message TEXT,
                    created_at TEXT DEFAULT (datetime('now','localtime')),
                    FOREIGN KEY (prediction_run_id) REFERENCES prediction_runs(id) ON DELETE CASCADE
                );
                """
            )
            self._ensure_prediction_outputs_schema(conn)
            self._ensure_training_runs_schema(conn)
            self._ensure_datasets_schema(conn)
            self._ensure_samples_schema(conn)
            self._ensure_models_schema(conn)

    def _ensure_prediction_outputs_schema(self, conn):
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(prediction_outputs)").fetchall()
        }
        if "overlay_path" not in columns:
            conn.execute("ALTER TABLE prediction_outputs ADD COLUMN overlay_path TEXT")
        if "iou" not in columns:
            conn.execute("ALTER TABLE prediction_outputs ADD COLUMN iou REAL")
        if "status" not in columns:
            conn.execute(
                "ALTER TABLE prediction_outputs "
                "ADD COLUMN status TEXT NOT NULL DEFAULT 'done'"
            )
        if "error_message" not in columns:
            conn.execute("ALTER TABLE prediction_outputs ADD COLUMN error_message TEXT")

    def _ensure_models_schema(self, conn):
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(models)").fetchall()
        }
        additions = {
            "parameter_count": "INTEGER",
            "db_import_test_mean_iou": "REAL",
            "db_import_test_n_samples": "INTEGER",
            "db_import_test_prediction_run_id": "INTEGER",
            "db_import_test_updated_at": "TEXT",
        }
        for column, column_type in additions.items():
            if column not in columns:
                conn.execute(f"ALTER TABLE models ADD COLUMN {column} {column_type}")

    def _ensure_training_runs_schema(self, conn):
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(training_runs)").fetchall()
        }
        if "duration_seconds" not in columns:
            conn.execute("ALTER TABLE training_runs ADD COLUMN duration_seconds REAL")

    def _ensure_datasets_schema(self, conn):
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(datasets)").fetchall()
        }
        if "storage_layout" not in columns:
            conn.execute(
                "ALTER TABLE datasets "
                "ADD COLUMN storage_layout TEXT NOT NULL DEFAULT 'legacy_nested'"
            )

    def _ensure_samples_schema(self, conn):
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(samples)").fetchall()
        }
        if "label_path" not in columns:
            conn.execute("ALTER TABLE samples ADD COLUMN label_path TEXT")

    @staticmethod
    def _dict(row):
        return dict(row) if row is not None else None

    @staticmethod
    def _json(value):
        return json.dumps(value, ensure_ascii=False)

    def upsert_source_excel(self, excel_path, output_name):
        path = os.path.abspath(excel_path)
        stat = os.stat(path)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id FROM source_excels
                WHERE original_path = ?
                ORDER BY updated_at DESC, id DESC
                LIMIT 1
                """,
                (path,),
            ).fetchone()
            if row:
                conn.execute(
                    """
                    UPDATE source_excels
                    SET file_name = ?,
                        output_name = ?,
                        file_mtime = ?,
                        file_size = ?,
                        status = 'active',
                        updated_at = datetime('now','localtime')
                    WHERE id = ?
                    """,
                    (os.path.basename(path), output_name, stat.st_mtime, stat.st_size, row["id"]),
                )
                return row["id"]

            cur = conn.execute(
                """
                INSERT INTO source_excels
                    (file_name, original_path, output_name, file_mtime, file_size, status)
                VALUES (?, ?, ?, ?, ?, 'active')
                """,
                (os.path.basename(path), path, output_name, stat.st_mtime, stat.st_size),
            )
        return cur.lastrowid

    def get_dataset_for_excel_path(self, excel_path, storage_layout=None):
        path = os.path.abspath(excel_path)
        params = [path]
        layout_filter = ""
        if storage_layout is not None:
            layout_filter = " AND d.storage_layout = ?"
            params.append(storage_layout)
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT d.*
                FROM datasets d
                JOIN source_excels s ON s.id = d.source_excel_id
                WHERE s.original_path = ?{layout_filter}
                ORDER BY
                    CASE WHEN d.status = 'done' THEN 0 ELSE 1 END,
                    d.updated_at DESC,
                    d.id DESC
                LIMIT 1
                """,
                params,
            ).fetchone()
        return self._dict(row)

    def get_dataset_for_source(self, source_excel_id, dataset_name):
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM datasets
                WHERE source_excel_id = ? AND dataset_name = ?
                """,
                (source_excel_id, dataset_name),
            ).fetchone()
        return self._dict(row)

    def upsert_dataset(
        self,
        dataset_name,
        source_excel_id,
        processed_dir,
        status="running",
        storage_layout="legacy_nested",
    ):
        processed_dir = os.path.abspath(processed_dir)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO datasets
                    (dataset_name, source_excel_id, processed_dir, storage_layout, status)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source_excel_id, dataset_name) DO UPDATE SET
                    processed_dir=excluded.processed_dir,
                    storage_layout=excluded.storage_layout,
                    status=excluded.status,
                    updated_at=datetime('now','localtime')
                """,
                (dataset_name, source_excel_id, processed_dir, storage_layout, status),
            )
            row = conn.execute(
                """
                SELECT id FROM datasets
                WHERE source_excel_id = ? AND dataset_name = ?
                """,
                (source_excel_id, dataset_name),
            ).fetchone()
        return row["id"]

    def replace_samples(self, dataset_id, samples):
        active_count = sum(
            1 for sample in samples if sample.get("status", "active") == "active"
        )
        dataset_status = "done" if active_count > 0 else "failed"
        with self._connect() as conn:
            conn.execute("DELETE FROM samples WHERE dataset_id = ?", (dataset_id,))
            conn.executemany(
                """
                INSERT INTO samples
                    (dataset_id, sample_name, image_path, ground_truth_path,
                     label_path, mask_view_path, width, height, status)
                VALUES
                    (:dataset_id, :sample_name, :image_path, :ground_truth_path,
                     :label_path, :mask_view_path, :width, :height, :status)
                """,
                [
                    dict(
                        sample,
                        dataset_id=dataset_id,
                        label_path=sample.get("label_path"),
                        status=sample.get("status", "active"),
                    )
                    for sample in samples
                ],
            )
            conn.execute(
                """
                UPDATE datasets
                SET n_samples = ?, status = ?, updated_at = datetime('now','localtime')
                WHERE id = ?
                """,
                (active_count, dataset_status, dataset_id),
            )
        return active_count

    def mark_dataset_failed(self, dataset_id):
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE datasets
                SET status = 'failed', updated_at = datetime('now','localtime')
                WHERE id = ?
                """,
                (dataset_id,),
            )

    def list_datasets(self, only_done=True, storage_layout=None):
        sql = """
            SELECT d.*, s.original_path
            FROM datasets d
            JOIN source_excels s ON s.id = d.source_excel_id
        """
        where = []
        params = []
        if only_done:
            where.append("d.status = ?")
            params.append("done")
        if storage_layout is not None:
            where.append("d.storage_layout = ?")
            params.append(storage_layout)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY d.updated_at DESC, d.id DESC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def get_dataset(self, dataset_id):
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        return self._dict(row)

    def list_samples(self, dataset_id):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM samples
                WHERE dataset_id = ? AND status = 'active'
                ORDER BY sample_name ASC
                """,
                (dataset_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def count_samples_by_status(self, dataset_id):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM samples
                WHERE dataset_id = ?
                GROUP BY status
                """,
                (dataset_id,),
            ).fetchall()
        counts = {"active": 0, "failed": 0}
        for row in rows:
            counts[row["status"]] = row["count"]
        return counts

    def cache_valid(self, dataset):
        if not dataset or dataset["status"] != "done" or dataset["n_samples"] <= 0:
            return False
        samples = self.list_samples(dataset["id"])
        if len(samples) != dataset["n_samples"]:
            return False
        for sample in samples:
            if not os.path.exists(sample["image_path"]):
                return False
            if not os.path.exists(sample["ground_truth_path"]):
                return False
        return True

    def create_split(
        self,
        dataset_id,
        split_name,
        train_ids,
        val_ids,
        test_ids=None,
        seed=42,
        train_ratio=0.8,
        val_ratio=0.2,
        test_ratio=0.0,
    ):
        test_ids = test_ids or []
        with self._connect() as conn:
            old = conn.execute(
                "SELECT id FROM dataset_splits WHERE dataset_id = ? AND split_name = ?",
                (dataset_id, split_name),
            ).fetchone()
            if old:
                split_id = old["id"]
                conn.execute("DELETE FROM sample_splits WHERE split_id = ?", (split_id,))
                conn.execute(
                    """
                    UPDATE dataset_splits
                    SET seed=?, train_ratio=?, val_ratio=?, test_ratio=?,
                        created_at=datetime('now','localtime')
                    WHERE id=?
                    """,
                    (seed, train_ratio, val_ratio, test_ratio, split_id),
                )
            else:
                cur = conn.execute(
                    """
                    INSERT INTO dataset_splits
                        (dataset_id, split_name, seed, train_ratio, val_ratio, test_ratio)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (dataset_id, split_name, seed, train_ratio, val_ratio, test_ratio),
                )
                split_id = cur.lastrowid
            rows = (
                [(split_id, sample_id, "train") for sample_id in train_ids]
                + [(split_id, sample_id, "val") for sample_id in val_ids]
                + [(split_id, sample_id, "test") for sample_id in test_ids]
            )
            conn.executemany(
                "INSERT INTO sample_splits (split_id, sample_id, split) VALUES (?, ?, ?)",
                rows,
            )
        return split_id

    def list_splits(self, dataset_id=None):
        sql = """
            SELECT sp.*, d.dataset_name
            FROM dataset_splits sp
            JOIN datasets d ON d.id = sp.dataset_id
        """
        params = []
        if dataset_id is not None:
            sql += " WHERE sp.dataset_id = ?"
            params.append(dataset_id)
        sql += " ORDER BY sp.created_at DESC, sp.id DESC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def get_split_samples(self, split_id, split):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.*
                FROM sample_splits ss
                JOIN samples s ON s.id = ss.sample_id
                WHERE ss.split_id = ? AND ss.split = ?
                ORDER BY s.sample_name ASC
                """,
                (split_id, split),
            ).fetchall()
        return [dict(row) for row in rows]

    def insert_training_run(
        self,
        run_name,
        dataset_id,
        split_id,
        encoder_name,
        decoder_name,
        epochs,
        batch_size,
        lr,
    ):
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO training_runs
                    (run_name, dataset_id, split_id, encoder_name, decoder_name,
                     epochs, batch_size, lr, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running')
                """,
                (run_name, dataset_id, split_id, encoder_name, decoder_name, epochs, batch_size, lr),
            )
        return cur.lastrowid

    def update_training_run(self, run_id, status=None, val_metric=None, duration_seconds=None):
        sets, params = [], []
        if status is not None:
            sets.append("status = ?")
            params.append(status)
        if val_metric is not None:
            sets.append("val_metric = ?")
            params.append(val_metric)
        if duration_seconds is not None:
            sets.append("duration_seconds = ?")
            params.append(float(duration_seconds))
        if not sets:
            return
        params.append(run_id)
        with self._connect() as conn:
            conn.execute(
                f"""
                UPDATE training_runs
                SET {', '.join(sets)}, updated_at = datetime('now','localtime')
                WHERE id = ?
                """,
                params,
            )

    def replace_training_run_params(self, run_id, params):
        with self._connect() as conn:
            conn.execute("DELETE FROM training_run_params WHERE run_id = ?", (run_id,))
            conn.executemany(
                """
                INSERT INTO training_run_params (run_id, param_name, param_value)
                VALUES (?, ?, ?)
                """,
                [
                    (run_id, str(key), self._json(value))
                    for key, value in sorted(params.items())
                ],
            )

    def get_training_run_params(self, run_id):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT param_name, param_value
                FROM training_run_params
                WHERE run_id = ?
                ORDER BY param_name ASC
                """,
                (run_id,),
            ).fetchall()
        return {row["param_name"]: row["param_value"] for row in rows}

    def replace_epoch_metrics(self, run_id, metrics):
        with self._connect() as conn:
            conn.execute("DELETE FROM training_epoch_metrics WHERE run_id = ?", (run_id,))
            conn.executemany(
                """
                INSERT INTO training_epoch_metrics
                    (run_id, epoch, train_loss, train_ap, train_iou, train_dice,
                     val_loss, val_ap, val_iou, val_dice, lr)
                VALUES
                    (:run_id, :epoch, :train_loss, :train_ap, :train_iou, :train_dice,
                     :val_loss, :val_ap, :val_iou, :val_dice, :lr)
                """,
                [dict(row, run_id=run_id) for row in metrics],
            )

    def upsert_epoch_metric(self, run_id, epoch, metrics):
        """Insert or replace one epoch row; for live training progress."""
        row = {
            "run_id": run_id,
            "epoch": int(epoch),
            "train_loss": metrics.get("train_loss"),
            "train_ap": metrics.get("train_ap"),
            "train_iou": metrics.get("train_iou"),
            "train_dice": metrics.get("train_dice"),
            "val_loss": metrics.get("val_loss"),
            "val_ap": metrics.get("val_ap"),
            "val_iou": metrics.get("val_iou"),
            "val_dice": metrics.get("val_dice"),
            "lr": metrics.get("lr"),
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO training_epoch_metrics
                    (run_id, epoch, train_loss, train_ap, train_iou, train_dice,
                     val_loss, val_ap, val_iou, val_dice, lr)
                VALUES
                    (:run_id, :epoch, :train_loss, :train_ap, :train_iou, :train_dice,
                     :val_loss, :val_ap, :val_iou, :val_dice, :lr)
                ON CONFLICT(run_id, epoch) DO UPDATE SET
                    train_loss = excluded.train_loss,
                    train_ap   = excluded.train_ap,
                    train_iou  = excluded.train_iou,
                    train_dice = excluded.train_dice,
                    val_loss   = excluded.val_loss,
                    val_ap     = excluded.val_ap,
                    val_iou    = excluded.val_iou,
                    val_dice   = excluded.val_dice,
                    lr         = excluded.lr
                """,
                row,
            )

    def insert_test_metrics(self, run_id, split_id, metrics):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO training_test_metrics
                    (run_id, split_id, n_samples, test_loss, test_iou, test_dice, test_ap)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    split_id,
                    metrics["n_samples"],
                    metrics.get("test_loss"),
                    metrics.get("test_iou"),
                    metrics.get("test_dice"),
                    metrics.get("test_ap"),
                ),
            )

    def list_test_metrics(self, run_id):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, run_id, split_id, n_samples,
                       test_loss, test_iou, test_dice, test_ap, created_at
                FROM training_test_metrics
                WHERE run_id = ?
                ORDER BY created_at DESC, id DESC
                """,
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def insert_model(
        self,
        run_id,
        model_dir,
        best_model_path=None,
        snapshot_path=None,
        opt_path=None,
        parameter_count=None,
    ):
        model_dir = os.path.abspath(os.fspath(model_dir))
        best_model_path = os.fspath(best_model_path) if best_model_path else None
        snapshot_path = os.fspath(snapshot_path) if snapshot_path else None
        opt_path = os.fspath(opt_path) if opt_path else None
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO models
                    (run_id, model_dir, best_model_path, snapshot_path, opt_path,
                     parameter_count)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    model_dir,
                    best_model_path,
                    snapshot_path,
                    opt_path,
                    parameter_count,
                ),
            )
        return cur.lastrowid

    def update_model_parameter_count(self, model_id, parameter_count):
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE models
                SET parameter_count = ?
                WHERE id = ?
                """,
                (int(parameter_count), model_id),
            )

    def update_model_db_import_metrics(
        self,
        model_id,
        mean_iou,
        n_samples,
        prediction_run_id,
    ):
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE models
                SET db_import_test_mean_iou = ?,
                    db_import_test_n_samples = ?,
                    db_import_test_prediction_run_id = ?,
                    db_import_test_updated_at = datetime('now','localtime')
                WHERE id = ?
                """,
                (mean_iou, int(n_samples), prediction_run_id, model_id),
            )

    def list_models(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT m.*, r.run_name, r.encoder_name, r.decoder_name, r.status AS run_status
                FROM models m
                JOIN training_runs r ON r.id = m.run_id
                ORDER BY m.id DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_training_run(self, run_id):
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM training_runs WHERE id = ?", (run_id,)).fetchone()
        return self._dict(row)

    def get_latest_training_run_id(self):
        """Return the max id of training_runs, or None if empty."""
        with self._connect() as conn:
            row = conn.execute("SELECT MAX(id) FROM training_runs").fetchone()
        return row[0] if row and row[0] is not None else None

    def list_epoch_metrics(self, run_id):
        """Return all epoch metrics for a run, ordered by epoch."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT epoch, train_loss, train_iou, train_dice,
                       val_loss, val_iou, val_dice, lr
                FROM training_epoch_metrics
                WHERE run_id = ?
                ORDER BY epoch ASC
                """,
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_model(self, model_id):
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT m.*, r.encoder_name, r.decoder_name, r.run_name, r.status AS run_status
                FROM models m
                JOIN training_runs r ON r.id = m.run_id
                WHERE m.id = ?
                """,
                (model_id,),
            ).fetchone()
        return self._dict(row)

    def list_db_import_samples_for_model(self, model_id):
        """Return active samples excluding the model run's train/val split samples."""
        with self._connect() as conn:
            model = conn.execute(
                """
                SELECT m.id AS model_id, r.split_id
                FROM models m
                JOIN training_runs r ON r.id = m.run_id
                WHERE m.id = ?
                """,
                (model_id,),
            ).fetchone()
            if model is None:
                return []
            split_id = model["split_id"]
            excluded_ids = set()
            if split_id is not None:
                excluded_ids = {
                    row["sample_id"]
                    for row in conn.execute(
                        """
                        SELECT sample_id
                        FROM sample_splits
                        WHERE split_id = ? AND split IN ('train', 'val')
                        """,
                        (split_id,),
                    ).fetchall()
                }
            rows = conn.execute(
                """
                SELECT s.*, d.dataset_name, d.storage_layout
                FROM samples s
                JOIN datasets d ON d.id = s.dataset_id
                WHERE s.status = 'active' AND d.status = 'done'
                ORDER BY d.dataset_name ASC, s.sample_name ASC
                """
            ).fetchall()
        return [dict(row) for row in rows if row["id"] not in excluded_ids]

    def insert_prediction_run(self, model_id, input_path, output_dir, mode):
        input_path = os.fspath(input_path)
        output_dir = os.fspath(output_dir)
        stored_input_path = (
            input_path
            if mode == "database" or input_path.startswith("db:")
            else os.path.abspath(input_path)
        )
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO prediction_runs
                    (model_id, input_path, output_dir, mode, status)
                VALUES (?, ?, ?, ?, 'running')
                """,
                (model_id, stored_input_path, os.path.abspath(output_dir), mode),
            )
        return cur.lastrowid

    def update_prediction_run(self, prediction_run_id, status):
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE prediction_runs
                SET status = ?, updated_at = datetime('now','localtime')
                WHERE id = ?
                """,
                (status, prediction_run_id),
            )

    def insert_prediction_output(
        self,
        prediction_run_id,
        image_path,
        mask_path,
        compare_path,
        overlay_path=None,
        iou=None,
        status="done",
        error_message=None,
    ):
        image_path = os.fspath(image_path)
        mask_path = os.fspath(mask_path)
        compare_path = os.fspath(compare_path)
        overlay_path = os.fspath(overlay_path) if overlay_path else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO prediction_outputs
                    (prediction_run_id, image_path, mask_path, overlay_path, compare_path,
                     iou, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction_run_id,
                    os.path.abspath(image_path),
                    os.path.abspath(mask_path) if mask_path else "",
                    os.path.abspath(overlay_path) if overlay_path else None,
                    os.path.abspath(compare_path) if compare_path else "",
                    iou,
                    status,
                    error_message,
                ),
            )
