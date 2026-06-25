import numpy as np

from load_clean_module import loader
from load_clean_module.loader import LoadCleanService


class FakeWorkbook:
    sheetnames = [loader.SHEET_IMAGE, loader.SHEET_LABEL]

    def __getitem__(self, key):
        return key


def test_process_excel_writes_typed_flat_layout(tmp_path, monkeypatch):
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    label = image.copy()

    def fake_extract_from_sheet(_sheet, key, data):
        data.setdefault("Pad:01/Left", {})[key] = image if key == "image" else label

    monkeypatch.setattr(loader.openpyxl, "load_workbook", lambda *_args, **_kwargs: FakeWorkbook())
    monkeypatch.setattr(loader, "extract_from_sheet", fake_extract_from_sheet)

    service = LoadCleanService(
        save_dir=tmp_path,
        max_ground_truth_ratio=1.0,
        min_ground_truth_pixels=0,
    )
    samples, violations = service._process_excel(str(tmp_path / "Source File.xlsx"), str(tmp_path))

    expected_name = "Source_File_Pad_01_Left.png"
    expected_paths = {
        "image_path": tmp_path / "image" / expected_name,
        "label_path": tmp_path / "label" / expected_name,
        "ground_truth_path": tmp_path / "ground_truth" / expected_name,
        "mask_view_path": tmp_path / "mask_view" / expected_name,
    }

    assert violations == []
    assert len(samples) == 1
    for key, path in expected_paths.items():
        assert samples[0][key] == str(path)
        assert path.exists()


def test_process_excel_rejects_sanitized_filename_collisions(tmp_path, monkeypatch):
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    label = image.copy()

    def fake_extract_from_sheet(_sheet, key, data):
        for sample_name in ("Pad:01", "Pad/01"):
            data.setdefault(sample_name, {})[key] = image if key == "image" else label

    monkeypatch.setattr(loader.openpyxl, "load_workbook", lambda *_args, **_kwargs: FakeWorkbook())
    monkeypatch.setattr(loader, "extract_from_sheet", fake_extract_from_sheet)

    service = LoadCleanService(
        save_dir=tmp_path,
        max_ground_truth_ratio=1.0,
        min_ground_truth_pixels=0,
    )

    try:
        service._process_excel(str(tmp_path / "Source.xlsx"), str(tmp_path))
    except ValueError as exc:
        assert "清洗輸出檔名重複" in str(exc)
    else:
        raise AssertionError("Expected duplicate sanitized filename to fail")
