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
