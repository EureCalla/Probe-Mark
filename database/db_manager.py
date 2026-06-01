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
                    created_at TEXT DEFAULT (datetime('now','localtime')),
                    FOREIGN KEY (run_id) REFERENCES training_runs(id)
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
                    compare_path TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now','localtime')),
                    FOREIGN KEY (prediction_run_id) REFERENCES prediction_runs(id) ON DELETE CASCADE
                );
                """
            )

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

    def get_dataset_for_excel_path(self, excel_path):
        path = os.path.abspath(excel_path)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT d.*
                FROM datasets d
                JOIN source_excels s ON s.id = d.source_excel_id
                WHERE s.original_path = ?
                ORDER BY
                    CASE WHEN d.status = 'done' THEN 0 ELSE 1 END,
                    d.updated_at DESC,
                    d.id DESC
                LIMIT 1
                """,
                (path,),
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

    def upsert_dataset(self, dataset_name, source_excel_id, processed_dir, status="running"):
        processed_dir = os.path.abspath(processed_dir)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO datasets
                    (dataset_name, source_excel_id, processed_dir, status)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source_excel_id, dataset_name) DO UPDATE SET
                    processed_dir=excluded.processed_dir,
                    status=excluded.status,
                    updated_at=datetime('now','localtime')
                """,
                (dataset_name, source_excel_id, processed_dir, status),
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
        with self._connect() as conn:
            conn.execute("DELETE FROM samples WHERE dataset_id = ?", (dataset_id,))
            conn.executemany(
                """
                INSERT INTO samples
                    (dataset_id, sample_name, image_path, ground_truth_path,
                     mask_view_path, width, height, status)
                VALUES
                    (:dataset_id, :sample_name, :image_path, :ground_truth_path,
                     :mask_view_path, :width, :height, 'active')
                """,
                [dict(sample, dataset_id=dataset_id) for sample in samples],
            )
            conn.execute(
                """
                UPDATE datasets
                SET n_samples = ?, status = 'done', updated_at = datetime('now','localtime')
                WHERE id = ?
                """,
                (len(samples), dataset_id),
            )

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

    def list_datasets(self, only_done=True):
        sql = """
            SELECT d.*, s.original_path
            FROM datasets d
            JOIN source_excels s ON s.id = d.source_excel_id
        """
        params = []
        if only_done:
            sql += " WHERE d.status = ?"
            params.append("done")
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

    def update_training_run(self, run_id, status=None, val_metric=None):
        sets, params = [], []
        if status is not None:
            sets.append("status = ?")
            params.append(status)
        if val_metric is not None:
            sets.append("val_metric = ?")
            params.append(val_metric)
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

    def insert_model(self, run_id, model_dir, best_model_path=None, snapshot_path=None, opt_path=None):
        model_dir = os.path.abspath(os.fspath(model_dir))
        best_model_path = os.fspath(best_model_path) if best_model_path else None
        snapshot_path = os.fspath(snapshot_path) if snapshot_path else None
        opt_path = os.fspath(opt_path) if opt_path else None
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO models
                    (run_id, model_dir, best_model_path, snapshot_path, opt_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, model_dir, best_model_path, snapshot_path, opt_path),
            )
        return cur.lastrowid

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

    def insert_prediction_run(self, model_id, input_path, output_dir, mode):
        input_path = os.fspath(input_path)
        output_dir = os.fspath(output_dir)
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO prediction_runs
                    (model_id, input_path, output_dir, mode, status)
                VALUES (?, ?, ?, ?, 'running')
                """,
                (model_id, os.path.abspath(input_path), os.path.abspath(output_dir), mode),
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

    def insert_prediction_output(self, prediction_run_id, image_path, mask_path, compare_path):
        image_path = os.fspath(image_path)
        mask_path = os.fspath(mask_path)
        compare_path = os.fspath(compare_path)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO prediction_outputs
                    (prediction_run_id, image_path, mask_path, compare_path)
                VALUES (?, ?, ?, ?)
                """,
                (
                    prediction_run_id,
                    os.path.abspath(image_path),
                    os.path.abspath(mask_path),
                    os.path.abspath(compare_path),
                ),
            )
