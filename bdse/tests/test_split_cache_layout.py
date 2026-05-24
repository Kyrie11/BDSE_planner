from pathlib import Path

from bdse.data.nuplan_dataset import PreprocessedBDSEDataset
from bdse.data.scenario_sampler import DBFileRecord, select_records


def test_train_subsplit_selects_exact_folder():
    records = [
        DBFileRecord(split="train", folder="train_1", path=Path("/tmp/train_1/a.db")),
        DBFileRecord(split="train", folder="train_2", path=Path("/tmp/train_2/b.db")),
        DBFileRecord(split="val", folder="val", path=Path("/tmp/val/c.db")),
    ]
    got = select_records(records, split="train_1")
    assert [r.folder for r in got] == ["train_1"]
    got_all = select_records(records, split="train")
    assert [r.folder for r in got_all] == ["train_1", "train_2"]


def test_preprocessed_train_loads_all_train_subfolders(tmp_path):
    p1 = tmp_path / "train_1" / "sampling_1" / "a.npz"
    p2 = tmp_path / "train_2" / "sampling_2" / "b.npz"
    pv = tmp_path / "val" / "sampling_v" / "c.npz"
    for p in (p1, p2, pv):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"placeholder")
    ds_train = PreprocessedBDSEDataset(tmp_path, split="train")
    assert ds_train.build_index() == [p1, p2]
    ds_one = PreprocessedBDSEDataset(tmp_path, split="train_1")
    assert ds_one.build_index() == [p1]
    ds_multi = PreprocessedBDSEDataset(tmp_path, split=["train_1", "train_2"])
    assert ds_multi.build_index() == [p1, p2]


def test_preprocessed_missing_concrete_split_does_not_scan_entire_root(tmp_path):
    p1 = tmp_path / "train_boston" / "log_a" / "a.npz"
    p1.parent.mkdir(parents=True, exist_ok=True)
    p1.write_bytes(b"placeholder")
    ds = PreprocessedBDSEDataset(tmp_path, split="train_pittsburgh")
    try:
        ds.build_index()
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing concrete split should not fall back to scanning the whole root")


def test_preprocessed_root_can_be_the_requested_split(tmp_path):
    root = tmp_path / "train_boston"
    p1 = root / "log_a" / "a.npz"
    p1.parent.mkdir(parents=True, exist_ok=True)
    p1.write_bytes(b"placeholder")
    ds = PreprocessedBDSEDataset(root, split="train_boston")
    assert ds.build_index() == [p1]


def test_resume_filename_index_finds_old_log_layout(tmp_path):
    from bdse.data.nuplan_dataset import DevkitScenarioIndexRecord, NuPlanBDSEDataset

    ds = NuPlanBDSEDataset.__new__(NuPlanBDSEDataset)
    ds.cfg = {"preprocess": {}}
    ds.split = "train_boston"
    ds.preprocessed_dir = tmp_path
    ds.records = []
    ds.use_devkit = True
    ds.num_workers = 1
    ds.use_process_pool = False
    ds._index = [DevkitScenarioIndexRecord(None, "train_boston", "train_boston", "new_log", "tok123", 70, 0)]
    old = tmp_path / "train_boston" / "old_log" / "tok123_it000070.npz"
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_bytes(b"placeholder")
    by_name = ds._build_resume_filename_index(tmp_path)
    assert by_name["tok123_it000070.npz"] == old
