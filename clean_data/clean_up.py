"""資料清洗 CLI：從 Excel 提取 image / label，輸出 ground_truth 並登記 SQLite。"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mpivr20_cms import get_clean_output_dir


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Probe-Mark 資料清洗")
    parser.add_argument(
        "--tasks-json",
        help='JSON 任務清單，格式：[{"excel": "...", "output_name": "..."}, ...]',
    )
    parser.add_argument(
        "--save-dir",
        default=get_clean_output_dir(),
        help="輸出資料夾，預設 mpivr20_cms.py 的共用資料路徑",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="強制重建快取，不沿用既有清洗結果",
    )
    args = parser.parse_args()

    tasks = None
    if args.tasks_json:
        with open(args.tasks_json, encoding="utf-8") as f:
            tasks = json.load(f)

    from load_clean_module import LoadCleanService

    service = LoadCleanService(tasks=tasks, save_dir=args.save_dir, force=args.force)
    service.run()


if __name__ == "__main__":
    main()
