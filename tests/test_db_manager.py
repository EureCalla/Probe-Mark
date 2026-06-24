from database import DBManager


def test_db_manager_dataset_sample_model_prediction_flow(tmp_path):
    db = DBManager(tmp_path / "probe_mark.db")
    db.init_db()

    excel_path = tmp_path / "source.xlsx"
    excel_path.write_text("fake excel", encoding="utf-8")

    source_id = db.upsert_source_excel(str(excel_path), "demo")
    dataset_id = db.upsert_dataset("demo", source_id, tmp_path / "processed" / "demo")

    sample_rows = []
    for sample_name in ("sample_001", "sample_002"):
        image = tmp_path / "processed" / "demo" / sample_name / "image.png"
        label = tmp_path / "processed" / "demo" / sample_name / "ground_truth.png"
        mask = tmp_path / "processed" / "demo" / sample_name / "mask_view.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"image")
        label.write_bytes(b"label")
        mask.write_bytes(b"mask")
        sample_rows.append(
            {
                "sample_name": sample_name,
                "image_path": str(image),
                "ground_truth_path": str(label),
                "mask_view_path": str(mask),
                "width": 10,
                "height": 20,
            }
        )

    db.replace_samples(
        dataset_id,
        sample_rows,
    )

    dataset = db.get_dataset(dataset_id)
    assert dataset["n_samples"] == 2
    assert db.cache_valid(dataset)
    assert db.get_dataset_for_excel_path(str(excel_path))["id"] == dataset_id

    sample_ids = [sample["id"] for sample in db.list_samples(dataset_id)]
    split_id = db.create_split(
        dataset_id,
        "default",
        train_ids=[sample_ids[0]],
        val_ids=[sample_ids[1]],
    )
    assert db.get_split_samples(split_id, "train")[0]["sample_name"] == "sample_001"

    run_id = db.insert_training_run(
        "run",
        dataset_id,
        split_id,
        "resnet18",
        "FPN",
        1,
        1,
        0.001,
    )
    db.update_training_run(run_id, status="done", val_metric=0.5)
    model_id = db.insert_model(run_id, tmp_path / "runs" / "run", "best_model.pt")
    assert db.get_model(model_id)["decoder_name"] == "FPN"

    prediction_run_id = db.insert_prediction_run(
        model_id,
        sample_rows[0]["image_path"],
        tmp_path / "predictions",
        "single",
    )
    db.insert_prediction_output(
        prediction_run_id,
        sample_rows[0]["image_path"],
        str(tmp_path / "predictions" / "sample_001_mask.png"),
        str(tmp_path / "predictions" / "sample_001_compare.png"),
    )
    db.update_prediction_run(prediction_run_id, "done")


def test_db_manager_filters_trainable_storage_layout(tmp_path):
    db = DBManager(tmp_path / "probe_mark.db")
    db.init_db()

    excel_path = tmp_path / "source.xlsx"
    excel_path.write_text("fake excel", encoding="utf-8")
    source_id = db.upsert_source_excel(str(excel_path), "demo")

    legacy_id = db.upsert_dataset("legacy", source_id, tmp_path / "processed" / "legacy")
    typed_id = db.upsert_dataset(
        "typed",
        source_id,
        tmp_path / "processed",
        storage_layout="typed_flat_v1",
    )
    for dataset_id, folder in ((legacy_id, "legacy"), (typed_id, "typed")):
        image = tmp_path / folder / "image.png"
        ground_truth = tmp_path / folder / "ground_truth.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"image")
        ground_truth.write_bytes(b"ground_truth")
        db.replace_samples(
            dataset_id,
            [
                {
                    "sample_name": folder,
                    "image_path": str(image),
                    "ground_truth_path": str(ground_truth),
                    "mask_view_path": None,
                    "width": 10,
                    "height": 20,
                }
            ],
        )

    trainable = db.list_datasets(only_done=True, storage_layout="typed_flat_v1")

    assert [dataset["id"] for dataset in trainable] == [typed_id]
    assert db.get_dataset(legacy_id)["storage_layout"] == "legacy_nested"
    assert db.get_dataset(typed_id)["storage_layout"] == "typed_flat_v1"
