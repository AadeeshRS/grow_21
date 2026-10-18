"""Model training routines for Grow21 knowledge-tracing benchmark.

Implements binary cross-entropy training with validation-based early stopping,
deterministic seed control, and uniform training protocols across DKT variants.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import DKTSequenceDataset
from .models import LogisticBaselineModel, ReliabilityGatedDKT, VanillaDKT
from .simulator import SimulatorEvent
from .tasks import TaskRegistry


def train_recurrent_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float = 0.003,
    weight_decay: float = 1e-4,
    max_epochs: int = 35,
    patience: int = 6,
    is_reliability_gated: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, List[float]], float]:
    """Trains a recurrent DKT model with validation-loss early stopping.

    Returns:
        (best_model, history_dict, elapsed_seconds)
    """
    if device is None:
        device = torch.device("cpu")

    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    bce_loss_fn = nn.BCELoss(reduction="none")

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    start_time = time.time()

    for epoch in range(max_epochs):
        model.train()
        total_train_loss = 0.0
        train_elements = 0

        for batch in train_loader:
            meta = batch["meta_features"].to(device)
            outcomes = batch["outcomes"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            if is_reliability_gated:
                reliabilities = batch["reliabilities"].to(device)
                preds, _ = model(meta, outcomes, reliabilities, mask=mask)
            else:
                preds, _ = model(meta, outcomes, mask=mask)

            # Masked BCE loss
            raw_loss = bce_loss_fn(preds, outcomes)
            loss = (raw_loss * mask).sum() / torch.clamp(mask.sum(), min=1.0)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer.step()

            total_train_loss += loss.item() * mask.sum().item()
            train_elements += int(mask.sum().item())

        epoch_train_loss = total_train_loss / max(1, train_elements)
        history["train_loss"].append(epoch_train_loss)

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        val_elements = 0

        with torch.no_grad():
            for batch in val_loader:
                meta = batch["meta_features"].to(device)
                outcomes = batch["outcomes"].to(device)
                mask = batch["mask"].to(device)

                if is_reliability_gated:
                    reliabilities = batch["reliabilities"].to(device)
                    preds, _ = model(meta, outcomes, reliabilities, mask=mask)
                else:
                    preds, _ = model(meta, outcomes, mask=mask)

                raw_loss = bce_loss_fn(preds, outcomes)
                total_val_loss += (raw_loss * mask).sum().item()
                val_elements += int(mask.sum().item())

        epoch_val_loss = total_val_loss / max(1, val_elements)
        history["val_loss"].append(epoch_val_loss)

        # Early stopping check
        if epoch_val_loss < best_val_loss - 1e-4:
            best_val_loss = epoch_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    elapsed_time = time.time() - start_time
    model.load_state_dict(best_weights)
    model.eval()
    return model, history, elapsed_time


def train_benchmark_models(
    train_seqs: List[List[SimulatorEvent]],
    val_seqs: List[List[SimulatorEvent]],
    task_registry: TaskRegistry,
    model_cfg: Dict[str, Any],
    seed: int = 42,
    selected_models: Optional[List[str]] = None,
    models_to_train: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Trains candidate models on identical learner-disjoint data based on selected_models.

    Returns:
        dict containing trained models, training histories, runtimes, and parameter counts.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    dkt_cfg = model_cfg.get("dkt", {})
    log_cfg = model_cfg.get("logistic", {})

    batch_size = int(dkt_cfg.get("batch_size", 64))
    hidden_dim = int(dkt_cfg.get("hidden_dim", 48))
    embedding_dim = int(dkt_cfg.get("embedding_dim", 24))
    lr = float(dkt_cfg.get("learning_rate", 0.003))
    patience = int(dkt_cfg.get("patience", 6))
    max_epochs = int(dkt_cfg.get("max_epochs", 35))

    candidate_models = models_to_train or selected_models or ["all"]
    if "all" in candidate_models:
        models_to_train_list = ["logistic_baseline", "vanilla_dkt", "reliability_gated_dkt"]
    else:
        models_to_train_list = list(candidate_models)
    models_to_train = models_to_train_list

    models_out: Dict[str, Any] = {}
    runtimes_out: Dict[str, float] = {}
    param_counts_out: Dict[str, int] = {}
    histories_out: Dict[str, Any] = {}

    MODEL_SEED_OFFSETS = {
        "logistic_baseline": 0,
        "vanilla_dkt": 100,
        "reliability_gated_dkt": 200,
    }

    # 1. Fit Logistic Regression Baseline
    if "logistic_baseline" in models_to_train:
        t0 = time.time()
        log_seed = seed + MODEL_SEED_OFFSETS["logistic_baseline"]
        log_model = LogisticBaselineModel(
            task_registry=task_registry,
            C=float(log_cfg.get("C", 1.0)),
            max_iter=int(log_cfg.get("max_iter", 1000)),
            random_state=log_seed,
        )
        log_model.fit(train_seqs, val_seqs)
        log_runtime = time.time() - t0
        models_out["logistic_baseline"] = log_model
        runtimes_out["logistic_baseline"] = round(log_runtime, 3)
        param_counts_out["logistic_baseline"] = log_model.count_parameters()

    # 2. PyTorch DataLoaders for DKT models if requested
    needs_dkt = any(m in models_to_train for m in ("vanilla_dkt", "reliability_gated_dkt"))
    if needs_dkt:
        train_dataset = DKTSequenceDataset(train_seqs, task_registry)
        val_dataset = DKTSequenceDataset(val_seqs, task_registry)

        sample_batch = next(iter(DataLoader(train_dataset, batch_size=batch_size, shuffle=False)))
        meta_dim = sample_batch["meta_features"].shape[-1]

        # 3. Train Vanilla DKT
        if "vanilla_dkt" in models_to_train:
            v_seed = seed + MODEL_SEED_OFFSETS["vanilla_dkt"]
            torch.manual_seed(v_seed)
            np.random.seed(v_seed)
            g_vanilla = torch.Generator()
            g_vanilla.manual_seed(v_seed)
            train_loader_v = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g_vanilla)
            val_loader_v = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            vanilla_model = VanillaDKT(
                meta_dim=meta_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
            )
            fitted_vanilla, vanilla_hist, vanilla_runtime = train_recurrent_model(
                model=vanilla_model,
                train_loader=train_loader_v,
                val_loader=val_loader_v,
                learning_rate=lr,
                max_epochs=max_epochs,
                patience=patience,
                is_reliability_gated=False,
            )
            models_out["vanilla_dkt"] = fitted_vanilla
            runtimes_out["vanilla_dkt"] = round(vanilla_runtime, 3)
            param_counts_out["vanilla_dkt"] = fitted_vanilla.count_parameters()
            histories_out["vanilla_dkt"] = vanilla_hist

        # 4. Train Reliability-Gated DKT
        if "reliability_gated_dkt" in models_to_train:
            g_seed = seed + MODEL_SEED_OFFSETS["reliability_gated_dkt"]
            torch.manual_seed(g_seed)
            np.random.seed(g_seed)
            g_gated = torch.Generator()
            g_gated.manual_seed(g_seed)
            train_loader_g = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g_gated)
            val_loader_g = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            gated_model = ReliabilityGatedDKT(
                meta_dim=meta_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
            )
            fitted_gated, gated_hist, gated_runtime = train_recurrent_model(
                model=gated_model,
                train_loader=train_loader_g,
                val_loader=val_loader_g,
                learning_rate=lr,
                max_epochs=max_epochs,
                patience=patience,
                is_reliability_gated=True,
            )
            models_out["reliability_gated_dkt"] = fitted_gated
            runtimes_out["reliability_gated_dkt"] = round(gated_runtime, 3)
            param_counts_out["reliability_gated_dkt"] = fitted_gated.count_parameters()
            histories_out["reliability_gated_dkt"] = gated_hist

    return {
        "models": models_out,
        "runtimes": runtimes_out,
        "param_counts": param_counts_out,
        "histories": histories_out,
    }

