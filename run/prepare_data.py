import shutil
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from tqdm import tqdm

from src.conf import PrepareDataConfig
from src.utils.common import trace

SERIES_SCHEMA = {
    "series_id": pl.Utf8,
    "step": pl.UInt32,
    "anglez": pl.Float32,
    "enmo": pl.Float32,
}


FEATURE_NAMES = [
    "anglez",
    "enmo",
    "step",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "minute_sin",
    "minute_cos",
    "anglez_sin",
    "anglez_cos",
]

ANGLEZ_MEAN = -8.810476
ANGLEZ_STD = 35.521877
ENMO_MEAN = 0.041315
ENMO_STD = 0.101829


def to_coord(x: pl.Expr, max_: int, name: str) -> list[pl.Expr]:
    rad = 2 * np.pi * (x % max_) / max_
    x_sin = rad.sin()
    x_cos = rad.cos()

    return [x_sin.alias(f"{name}_sin"), x_cos.alias(f"{name}_cos")]


def deg_to_rad(x: pl.Expr) -> pl.Expr:
    return np.pi / 180 * x

def add_rolling_features(x: pl.Expr, rolling_steps: int, name: str) -> list[pl.Expr]:
    # Center = True setting the labels at the center of the window, otherwise the window is forward-moving
    
    features = [x.rolling_mean(rolling_steps, center=True, min_periods=1).abs().cast(pl.Float32).alias(f'{name}_rolling_abs_mean')]
    features += [x.rolling_max(rolling_steps, center=True, min_periods=1).abs().cast(pl.Float32).alias(f'{name}_rolling_abs_max')]
    features += [x.rolling_min(rolling_steps, center=True, min_periods=1).abs().cast(pl.Float32).alias(f'{name}_rolling_abs_min')]
    features += [x.diff().abs().rolling_std(rolling_steps, center=True, min_periods=1).cast(pl.Float32).alias(f'{name}_rolling_abs_diff_std')]
    features += [x.diff().abs().rolling_mean(rolling_steps, center=True, min_periods=1).cast(pl.Float32).alias(f'{name}_rolling_abs_diff_mean')]

    return features

def add_feature(series_df: pl.DataFrame) -> pl.DataFrame:

    signal_awake_pl = pl.from_records([list(range(1440)), list(np.sin(np.linspace(0, np.pi, 1440) + 0.208 * np.pi) ** 24)], schema = ["hour_minute", "signal_awake"])
    signal_onset_pl = pl.from_records([list(range(1440)), list(np.sin(np.linspace(0, np.pi, 1440) + 0.555 * np.pi) ** 24)], schema = ["hour_minute", "signal_onset"])

    series_df = (
        series_df
        # .with_row_count("step")
        .with_columns(pl.col("timestamp").dt.hour().alias("hour"))
        .with_columns(pl.col("timestamp").dt.minute().alias("minute"))
        .with_columns(deg_to_rad(pl.col("anglez")).alias("anglez_rad"))
        .with_columns(
            *to_coord(pl.col("timestamp").dt.hour(), 24, "hour"),
            *to_coord(pl.col("timestamp").dt.month(), 12, "month"),
            *to_coord(pl.col("timestamp").dt.minute(), 60, "minute"),
            # (pl.col("step") / pl.count("step")).alias("step_pct"),
            pl.col('anglez_rad').sin().alias('anglez_sin'),
            pl.col('anglez_rad').cos().alias('anglez_cos')
        )
        .with_columns(((pl.col("enmo") - pl.col("enmo").mean().over(["series_id"])) / pl.col("enmo").std().over(["series_id"])).alias("standardized_enmo"))
        .with_columns(
            *add_rolling_features(pl.col("anglez"), 60, "anglez"),
            *add_rolling_features(pl.col("enmo"), 60, "enmo"),
            (pl.col("hour")*60 + pl.col("minute")).cast(pl.Int64).alias("hour_minute"),
            (pl.col("enmo").diff() == 0).cast(pl.Int8).alias("is_enmo_diff_zero"),
            (pl.col("anglez").diff() == 0).cast(pl.Int8).alias("is_anglez_diff_zero")
        )
        .lazy().join(signal_awake_pl.select(["hour_minute", "signal_awake"]).lazy(), on = ['hour_minute'], how = "left")
        .lazy().join(signal_onset_pl.select(["hour_minute", "signal_onset"]).lazy(), on = ['hour_minute'], how = "left")
        .with_columns(
            pl.col("is_enmo_diff_zero").rolling_sum(120, center=True, min_periods=1).cast(pl.Float32).alias('rolling_sum_enmo_diff_zero'),
            pl.col("is_anglez_diff_zero").rolling_sum(120, center=True, min_periods=1).cast(pl.Float32).alias('rolling_sum_anglez_diff_zero')
        )
    )
    return series_df


def save_each_series(this_series_df: pl.DataFrame, columns: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for col_name in columns:
        x = this_series_df.get_column(col_name).to_numpy(zero_copy_only=True)
        np.save(output_dir / f"{col_name}.npy", x)


@hydra.main(config_path="conf", config_name="prepare_data", version_base="1.2")
def main(cfg: PrepareDataConfig):
    processed_dir: Path = Path(cfg.dir.processed_dir) / cfg.phase

    # ディレクトリが存在する場合は削除
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
        print(f"Removed {cfg.phase} dir: {processed_dir}")

    with trace("Load series"):
        # scan parquet
        if cfg.phase in ["train", "test"]:
            series_lf = pl.scan_parquet(
                Path(cfg.dir.data_dir) / f"{cfg.phase}_series.parquet",
                low_memory=True,
            )
        elif cfg.phase == "dev":
            series_lf = pl.scan_parquet(
                Path(cfg.dir.processed_dir) / f"{cfg.phase}_series.parquet",
                low_memory=True,
            )
        else:
            raise ValueError(f"Invalid phase: {cfg.phase}")

        # preprocess
        series_df = (
            series_lf.with_columns(
                pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"),
                deg_to_rad(pl.col("anglez")).alias("anglez_rad"),
                (pl.col("anglez") - ANGLEZ_MEAN) / ANGLEZ_STD,
                (pl.col("enmo") - ENMO_MEAN) / ENMO_STD,
            )
            .select(
                [
                    pl.col("series_id"),
                    pl.col("anglez"),
                    pl.col("enmo"),
                    pl.col("timestamp"),
                    pl.col("anglez_rad"),
                ]
            )
            .collect(streaming=True)
            .sort(by=["series_id", "timestamp"])
        )
        n_unique = series_df.get_column("series_id").n_unique()
    with trace("Save features"):
        for series_id, this_series_df in tqdm(series_df.group_by("series_id"), total=n_unique):
            # 特徴量を追加
            this_series_df = add_feature(this_series_df)

            # 特徴量をそれぞれnpyで保存
            series_dir = processed_dir / series_id  # type: ignore
            save_each_series(this_series_df, FEATURE_NAMES, series_dir)


if __name__ == "__main__":
    main()
