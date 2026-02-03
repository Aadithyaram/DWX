"""Train an autoencoder to estimate feature weightages for selected metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _read_metrics_list(metrics_arg: str | None) -> List[str]:
    if not metrics_arg:
        return []

    metrics_path = Path(metrics_arg)
    if metrics_path.exists():
        raw = metrics_path.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw.strip("[]")
        return [item.strip() for item in raw.split(",") if item.strip()]

    return [item.strip() for item in metrics_arg.split(",") if item.strip()]


def _load_excel_data(excel_path: Path, metrics: Sequence[str]) -> pd.DataFrame:
    data = pd.read_excel(excel_path)
    if metrics:
        missing = [metric for metric in metrics if metric not in data.columns]
        if missing:
            raise ValueError(
                "Metrics not found in Excel data: " + ", ".join(sorted(missing))
            )
        data = data[list(metrics)]
    return data


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        dropout = 0.1
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(16, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(16, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(16, input_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(inputs)
        return self.decoder(latent)


def _train_autoencoder(
    train_features: np.ndarray,
    val_features: np.ndarray,
    test_features: np.ndarray,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
    patience: int,
    min_delta: float,
    weight_decay: float,
    grad_clip: float,
) -> tuple[AutoEncoder, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = train_features.shape[1]
    model = AutoEncoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = nn.SmoothL1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    train_dataset = TensorDataset(torch.from_numpy(train_features).float())
    val_dataset = TensorDataset(torch.from_numpy(val_features).float())
    test_dataset = TensorDataset(torch.from_numpy(test_features).float())
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    best_state = None
    best_val = float("inf")
    epochs_without_improve = 0

    for _ in range(epochs):
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                reconstruction = model(batch)
                val_losses.append(loss_fn(reconstruction, batch).item())
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        scheduler.step(val_loss)

        if best_val - val_loss > min_delta:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    test_losses = []
    with torch.no_grad():
        for (batch,) in test_loader:
            batch = batch.to(device)
            reconstruction = model(batch)
            test_losses.append(loss_fn(reconstruction, batch).item())
    test_loss = float(np.mean(test_losses)) if test_losses else float("inf")

    return model, {"validation_loss": best_val, "test_loss": test_loss}


def _normalize_importances(
    importances: np.ndarray, feature_names: Iterable[str]
) -> dict[str, float]:
    total = float(np.sum(importances))
    if math.isclose(total, 0.0):
        normalized = np.zeros_like(importances)
    else:
        normalized = importances / total
    paired = [
        (feature, float(weight))
        for feature, weight in zip(feature_names, normalized, strict=False)
    ]
    paired.sort(key=lambda item: item[1], reverse=True)
    return dict(paired)


def _prepare_features(
    train_data: pd.DataFrame,
    eval_data: pd.DataFrame | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    StandardScaler | RobustScaler,
    dict[str, PowerTransformer | None],
    list[str],
]:
    numeric_train = train_data.apply(pd.to_numeric, errors="coerce")
    numeric_eval = eval_data.apply(pd.to_numeric, errors="coerce") if eval_data is not None else None
    numeric_train = _replace_sentinel_values(numeric_train)
    if numeric_eval is not None:
        numeric_eval = _replace_sentinel_values(numeric_eval)
    numeric_train, numeric_eval, _ = _add_missing_flags(numeric_train, numeric_eval)
    numeric_train, numeric_eval = _impute_missing(numeric_train, numeric_eval)
    numeric_train, numeric_eval, transformers = _apply_column_transforms(
        numeric_train, numeric_eval
    )
    if numeric_train.empty:
        raise ValueError("No numeric rows available after cleaning the data.")
    scaler: StandardScaler | RobustScaler = (
        RobustScaler() if _has_heavy_outliers(numeric_train) else StandardScaler()
    )
    scaled_train = scaler.fit_transform(numeric_train.values)
    scaled_eval = (
        scaler.transform(numeric_eval.values) if numeric_eval is not None else None
    )
    feature_names = list(numeric_train.columns)
    return (
        scaled_train.astype(np.float32),
        scaled_eval.astype(np.float32) if scaled_eval is not None else None,
        scaler,
        transformers,
        feature_names,
    )


def _validate_splits(validation_split: float, test_split: float) -> None:
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be between 0 and 1 (exclusive).")
    if not 0.0 < test_split < 1.0:
        raise ValueError("test_split must be between 0 and 1 (exclusive).")
    if validation_split + test_split >= 1.0:
        raise ValueError("validation_split + test_split must be less than 1.")


def _split_dataframe(
    data: pd.DataFrame,
    validation_split: float,
    test_split: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(data))
    val_size = int(len(data) * validation_split)
    test_size = int(len(data) * test_split)
    train_size = len(data) - val_size - test_size
    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]
    return (
        data.iloc[train_idx].reset_index(drop=True),
        data.iloc[val_idx].reset_index(drop=True),
        data.iloc[test_idx].reset_index(drop=True),
    )


def _replace_sentinel_values(data: pd.DataFrame) -> pd.DataFrame:
    sentinel_values = {2147483647, 9.223372e18}
    return data.replace(list(sentinel_values), np.nan)


def _impute_missing(
    train_data: pd.DataFrame, eval_data: pd.DataFrame | None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    medians = train_data.median()
    train_imputed = train_data.fillna(medians)
    eval_imputed = eval_data.fillna(medians) if eval_data is not None else None
    return train_imputed, eval_imputed


def _add_missing_flags(
    train_data: pd.DataFrame, eval_data: pd.DataFrame | None, threshold: float = 0.05
) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
    missing_flags: list[str] = []
    train_missing = train_data.isna().mean()
    flagged = train_missing[train_missing > threshold].index
    for column in flagged:
        flag_name = f"{column}__missing"
        missing_flags.append(flag_name)
        train_data[flag_name] = train_data[column].isna().astype(float)
        if eval_data is not None:
            eval_data[flag_name] = eval_data[column].isna().astype(float)
    return train_data, eval_data, missing_flags


def _apply_column_transforms(
    train_data: pd.DataFrame, eval_data: pd.DataFrame | None
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, PowerTransformer | None]]:
    transformers: dict[str, PowerTransformer | None] = {}
    log_columns = {
        "freeStorageSpaceInBytes",
        "readIOPS",
        "writeIOPS",
        "tcpDataReceivedMB",
        "tcpDataSentMB",
    }
    for column in train_data.columns:
        if column.endswith("__missing"):
            transformers[column] = None
            continue
        series = train_data[column]
        if column in log_columns:
            train_data[column] = np.log1p(series.clip(lower=0))
            if eval_data is not None:
                eval_data[column] = np.log1p(eval_data[column].clip(lower=0))
            transformers[column] = None
            continue
        skewness = series.skew()
        if abs(skewness) > 1.0:
            transformer = PowerTransformer(method="yeo-johnson", standardize=False)
            train_data[column] = transformer.fit_transform(series.to_frame()).ravel()
            if eval_data is not None:
                eval_data[column] = transformer.transform(
                    eval_data[column].to_frame()
                ).ravel()
            transformers[column] = transformer
        else:
            transformers[column] = None
    return train_data, eval_data, transformers


def _has_heavy_outliers(data: pd.DataFrame) -> bool:
    quantiles = data.quantile([0.25, 0.75])
    iqr = quantiles.loc[0.75] - quantiles.loc[0.25]
    if (iqr == 0).all():
        return False
    high_outliers = (data > (quantiles.loc[0.75] + 3 * iqr)).any()
    return bool(high_outliers.any())


def _compute_permutation_importance(
    model: AutoEncoder,
    features: np.ndarray,
    feature_names: Sequence[str],
    repeats: int,
    seed: int,
    device: str,
) -> tuple[dict[str, float], dict[str, float]]:
    model.eval()
    loss_fn = nn.SmoothL1Loss()
    rng = np.random.default_rng(seed)
    base_loss = _reconstruction_loss(model, features, loss_fn, device)
    importances = np.zeros(features.shape[1], dtype=np.float64)
    importances_std = np.zeros(features.shape[1], dtype=np.float64)
    for idx in range(features.shape[1]):
        losses = []
        for _ in range(repeats):
            shuffled = features.copy()
            rng.shuffle(shuffled[:, idx])
            loss = _reconstruction_loss(model, shuffled, loss_fn, device)
            losses.append(loss - base_loss)
        importances[idx] = float(np.mean(losses))
        importances_std[idx] = float(np.std(losses))
    return (
        _normalize_importances(importances, feature_names),
        _normalize_importances(importances_std, feature_names),
    )


def _compute_group_permutation_importance(
    model: AutoEncoder,
    features: np.ndarray,
    groups: dict[str, list[int]],
    repeats: int,
    seed: int,
    device: str,
) -> tuple[dict[str, float], dict[str, float]]:
    model.eval()
    loss_fn = nn.SmoothL1Loss()
    rng = np.random.default_rng(seed)
    base_loss = _reconstruction_loss(model, features, loss_fn, device)
    group_names = list(groups.keys())
    importances = np.zeros(len(group_names), dtype=np.float64)
    importances_std = np.zeros(len(group_names), dtype=np.float64)
    for idx, name in enumerate(group_names):
        losses = []
        group_indices = groups[name]
        for _ in range(repeats):
            shuffled = features.copy()
            for column_idx in group_indices:
                rng.shuffle(shuffled[:, column_idx])
            loss = _reconstruction_loss(model, shuffled, loss_fn, device)
            losses.append(loss - base_loss)
        importances[idx] = float(np.mean(losses))
        importances_std[idx] = float(np.std(losses))
    return (
        _normalize_importances(importances, group_names),
        _normalize_importances(importances_std, group_names),
    )


def _reconstruction_loss(
    model: AutoEncoder, features: np.ndarray, loss_fn: nn.Module, device: str
) -> float:
    dataset = TensorDataset(torch.from_numpy(features).float())
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    losses = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            reconstruction = model(batch)
            losses.append(loss_fn(reconstruction, batch).item())
    return float(np.mean(losses)) if losses else float("inf")


def _build_feature_groups(feature_names: Sequence[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    zscaler_indices = [
        idx for idx, name in enumerate(feature_names) if "zscaler" in name.lower()
    ]
    if zscaler_indices:
        groups["zscaler"] = zscaler_indices
    iops_indices = [
        idx
        for idx, name in enumerate(feature_names)
        if name in {"readIOPS", "writeIOPS"}
    ]
    if iops_indices:
        groups["iops"] = iops_indices
    tcp_indices = [
        idx
        for idx, name in enumerate(feature_names)
        if name in {"tcpDataReceivedMB", "tcpDataSentMB"}
    ]
    if tcp_indices:
        groups["tcp"] = tcp_indices
    return groups


def _summarize_importances(
    importances: list[dict[str, float]],
    feature_names: Sequence[str],
) -> tuple[dict[str, float], dict[str, float]]:
    matrix = np.array(
        [[importance.get(name, 0.0) for name in feature_names] for importance in importances]
    )
    mean_vals = matrix.mean(axis=0)
    std_vals = matrix.std(axis=0)
    return (
        _normalize_importances(mean_vals, feature_names),
        _normalize_importances(std_vals, feature_names),
    )


def _top_k_overlap(weightage_sets: list[dict[str, float]], k: int = 5) -> float:
    if not weightage_sets:
        return 0.0
    top_sets = [
        set(list(weights.keys())[:k]) for weights in weightage_sets if weights
    ]
    if not top_sets:
        return 0.0
    intersection = set.intersection(*top_sets)
    return len(intersection) / k


def run(
    excel_path: Path,
    metrics: Sequence[str],
    output_path: Path,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
    validation_split: float,
    test_split: float,
    patience: int,
    min_delta: float,
    num_seeds: int,
    permutation_repeats: int,
) -> dict[str, float]:
    data = _load_excel_data(excel_path, metrics)
    _validate_splits(validation_split, test_split)
    train_data, val_data, test_data = _split_dataframe(
        data, validation_split, test_split, seed
    )
    train_features, val_features, _, _, feature_names = _prepare_features(
        train_data, val_data
    )
    _, test_features, _, _, _ = _prepare_features(train_data, test_data)
    if train_features is None or val_features is None or test_features is None:
        raise ValueError("Failed to build training/validation/test feature sets.")
    if latent_dim not in {2, 4, 8}:
        raise ValueError("latent_dim must be one of 2, 4, or 8.")
    if latent_dim >= train_features.shape[1]:
        raise ValueError("latent_dim must be smaller than the number of features.")
    sample_count = train_features.shape[0]
    batch_size = min(batch_size, max(1, sample_count))
    num_seeds = max(1, min(num_seeds, 5))

    weightage_runs: list[dict[str, float]] = []
    group_weightage_runs: list[dict[str, float]] = []
    groups = _build_feature_groups(feature_names)

    for offset in range(num_seeds):
        run_seed = seed + offset
        model, _ = _train_autoencoder(
            train_features=train_features,
            val_features=val_features,
            test_features=test_features,
            latent_dim=latent_dim,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=run_seed,
            device=device,
            patience=patience,
            min_delta=min_delta,
            weight_decay=1e-4,
            grad_clip=1.0,
        )

        weightages, _ = _compute_permutation_importance(
            model=model,
            features=val_features,
            feature_names=feature_names,
            repeats=permutation_repeats,
            seed=run_seed,
            device=device,
        )
        weightage_runs.append(weightages)
        if groups:
            group_weightages, _ = _compute_group_permutation_importance(
                model=model,
                features=val_features,
                groups=groups,
                repeats=permutation_repeats,
                seed=run_seed,
                device=device,
            )
            group_weightage_runs.append(group_weightages)

    weightage_mean, weightage_std = _summarize_importances(
        weightage_runs, feature_names
    )
    group_mean: dict[str, float] = {}
    group_std: dict[str, float] = {}
    if group_weightage_runs and groups:
        group_names = list(groups.keys())
        group_mean, group_std = _summarize_importances(
            group_weightage_runs, group_names
        )

    output_payload = {
        "weightages_mean": weightage_mean,
        "weightages_std": weightage_std,
        "group_weightages_mean": group_mean,
        "group_weightages_std": group_std,
        "top5_overlap": _top_k_overlap(weightage_runs, k=5),
    }
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    return weightage_mean


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train an autoencoder on selected Excel metrics and output feature weightages."
        )
    )
    parser.add_argument("--excel", required=True, type=Path, help="Path to Excel file.")
    parser.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated metrics or path to a JSON/TXT list of metrics.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("weightages.json"),
        help="Output JSON path for weightages.",
    )
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of data reserved for validation.",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Fraction of data reserved for testing.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Epochs to wait for validation improvement before early stopping.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation loss improvement to reset patience.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=3,
        help="Number of training runs with different seeds (max 5).",
    )
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=20,
        help="Number of shuffles per feature/group for permutation importance.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for training.",
    )

    args = parser.parse_args()
    metrics = _read_metrics_list(args.metrics)
    run(
        excel_path=args.excel,
        metrics=metrics,
        output_path=args.output,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        validation_split=args.validation_split,
        test_split=args.test_split,
        patience=args.patience,
        min_delta=args.min_delta,
        num_seeds=args.num_seeds,
        permutation_repeats=args.permutation_repeats,
    )


if __name__ == "__main__":
    main()
