import numpy as np
import pandas as pd
import pytest

from stair_common import training


def history():
    rows = []
    for seed in (0, 1, 2):
        for epoch, (tr, va) in enumerate([(0.9, 0.80), (0.6, 0.62), (0.4, 0.60), (0.2, 0.70)], start=1):
            rows.append({"track": "systemOne", "model": "judge", "run": "qpa", "seed": seed, "epoch": epoch,
                         "train_loss": tr, "val_loss": va + 0.01 * seed})
    return pd.DataFrame(rows)


def test_validate_accepts_the_layout():
    training.validate_history(history())


@pytest.mark.parametrize("break_it", [
    lambda df: df.drop(columns="val_loss"),
    lambda df: df.iloc[::-1],
    lambda df: df.assign(train_loss=df["train_loss"].where(df.index != 2, np.nan)),
    lambda df: pd.concat([df, df.iloc[:1]]),
])
def test_validate_rejects(break_it):
    with pytest.raises(ValueError):
        training.validate_history(break_it(history()))


def test_best_epoch_is_lowest_val_loss():
    best = training.best_epoch(history())
    assert len(best) == 3 and (best["epoch"] == 3).all()
    assert len(training.best_epoch(pd.concat([history(), history().assign(run="qp")]))) == 6  # repeated index


def test_from_records_maps_dev_nll():
    records = [{"epoch": 1, "train_loss": 0.7, "dev_nll": 0.6, "seconds": 3.0},
               {"epoch": 2, "train_loss": 0.5, "dev_nll": 0.55, "seconds": 3.1}]
    df = training.from_records(records, "systemOne", "nano-jev judge", "qpa", 0, val_key="dev_nll")
    training.validate_history(df)
    assert list(df.columns) == training.HISTORY_COLUMNS and df["val_loss"].tolist() == [0.6, 0.55]


def test_plot_writes_a_file(tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "curves.png"
    training.plot_history(history(), out)
    assert out.stat().st_size > 0
