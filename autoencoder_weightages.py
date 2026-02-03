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
from torch.utils.data import DataLoader, TensorDataset, random_split


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
        hidden_dims = _select_hidden_dims(input_dim, latent_dim)
        encoder_layers: list[nn.Module] = []
        in_features = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(in_features, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=0.1),
                ]
            )
            in_features = hidden_dim
        encoder_layers.append(nn.Linear(in_features, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        in_features = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(in_features, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=0.1),
                ]
            )
            in_features = hidden_dim
        decoder_layers.append(nn.Linear(in_features, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(inputs)
        return self.decoder(latent)


def _train_autoencoder(
    features: np.ndarray,
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
) -> tuple[AutoEncoder, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = features.shape[1]
    model = AutoEncoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(torch.from_numpy(features).float())
    val_size = max(1, int(len(dataset) * validation_split))
    test_size = max(1, int(len(dataset) * test_split))
    train_size = max(1, len(dataset) - val_size - test_size)
    if train_size + val_size + test_size > len(dataset):
        test_size = max(1, len(dataset) - train_size - val_size)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                reconstruction = model(batch)
                val_losses.append(loss_fn(reconstruction, batch).item())
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

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


def _feature_weightages(model: AutoEncoder, feature_names: Iterable[str]) -> dict[str, float]:
    encoder_layer = model.encoder[0]
    weights = encoder_layer.weight.detach().cpu().numpy()
    importance = np.mean(np.abs(weights), axis=0)
    total = float(np.sum(importance))
    if math.isclose(total, 0.0):
        normalized = np.zeros_like(importance)
    else:
        normalized = importance / total
    paired = [
        (feature, float(weight))
        for feature, weight in zip(feature_names, normalized, strict=False)
    ]
    paired.sort(key=lambda item: item[1], reverse=True)
    return dict(paired)


def _prepare_features(
    data: pd.DataFrame,
) -> tuple[np.ndarray, StandardScaler | RobustScaler, PowerTransformer | None]:
    numeric_data = data.apply(pd.to_numeric, errors="coerce")
    numeric_data = numeric_data.dropna()
    if numeric_data.empty:
        raise ValueError("No numeric rows available after cleaning the data.")
    skewness = numeric_data.skew().abs().max()
    power_transformer = None
    values = numeric_data.values
    if skewness > 1.0:
        power_transformer = PowerTransformer(method="yeo-johnson", standardize=False)
        values = power_transformer.fit_transform(values)
        scaler: StandardScaler | RobustScaler = RobustScaler()
    else:
        scaler = StandardScaler()
    scaled = scaler.fit_transform(values)
    return scaled.astype(np.float32), scaler, power_transformer


def _select_hidden_dims(input_dim: int, latent_dim: int) -> list[int]:
    if input_dim <= 4:
        return [max(latent_dim + 1, input_dim)]
    if input_dim <= 16:
        return [max(input_dim // 2, latent_dim + 2)]
    first = max(input_dim // 2, latent_dim + 4)
    second = max(input_dim // 4, latent_dim + 2)
    if second >= first:
        second = max(latent_dim + 1, first - 1)
    return [first, second]


def _validate_splits(validation_split: float, test_split: float) -> None:
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be between 0 and 1 (exclusive).")
    if not 0.0 < test_split < 1.0:
        raise ValueError("test_split must be between 0 and 1 (exclusive).")
    if validation_split + test_split >= 1.0:
        raise ValueError("validation_split + test_split must be less than 1.")


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
) -> dict[str, float]:
    data = _load_excel_data(excel_path, metrics)
    features, _, _ = _prepare_features(data)
    if latent_dim <= 0:
        raise ValueError("latent_dim must be a positive integer.")
    if latent_dim >= features.shape[1]:
        latent_dim = max(1, features.shape[1] // 2)
    if latent_dim == 2 and features.shape[1] > 4:
        latent_dim = min(8, max(2, features.shape[1] // 2))
    _validate_splits(validation_split, test_split)
    sample_count = features.shape[0]
    if sample_count < 200:
        epochs = max(50, min(epochs, 300))
    elif sample_count > 5000:
        epochs = min(epochs, 150)
    batch_size = min(batch_size, max(1, min(128, sample_count)))

    model, _ = _train_autoencoder(
        features=features,
        latent_dim=latent_dim,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
        validation_split=validation_split,
        test_split=test_split,
        patience=patience,
        min_delta=min_delta,
    )

    weightages = _feature_weightages(model, data.columns)
    output_path.write_text(json.dumps(weightages, indent=2), encoding="utf-8")
    return weightages


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
    parser.add_argument("--latent-dim", type=int, default=2)
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
    )


if __name__ == "__main__":
    main()
