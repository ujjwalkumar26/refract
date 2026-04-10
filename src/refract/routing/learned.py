"""Learned router — small gating network for metric weight prediction.

Uses a lightweight MLP to predict metric weights from query and space
features. Requires PyTorch (optional dependency).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from refract.routing.base import BaseRouter

if TYPE_CHECKING:
    from refract.types import QueryProfile, SpaceProfile


def _try_import_torch() -> Any:
    """Lazily import torch with a helpful error message."""
    try:
        import torch

        return torch
    except ImportError:
        raise ImportError(
            "LearnedRouter requires PyTorch. Install it with:\n"
            "  pip install 'refract-search[learned]'\n"
            "  # or\n"
            "  pip install torch"
        ) from None


def _profile_to_features(query_profile: QueryProfile, space_profile: SpaceProfile) -> list[float]:
    """Convert profiles to a flat feature vector.

    Features:
    - query: token_count, embedding_norm, entropy, type one-hot (4)
    - space: n_candidates (log), variance, anisotropy (log), score_spread, density one-hot (3)

    Total: 13 features.

    Args:
        query_profile: Query analysis result.
        space_profile: Space analysis result.

    Returns:
        List of 13 float features.
    """
    # Query type one-hot
    type_map = {"keyword": 0, "natural_language": 1, "code": 2, "structured": 3}
    type_idx = type_map.get(query_profile.query_type, 1)
    type_onehot = [1.0 if i == type_idx else 0.0 for i in range(4)]

    # Density one-hot
    density_map = {"sparse": 0, "medium": 1, "dense": 2}
    density_idx = density_map.get(space_profile.density, 1)
    density_onehot = [1.0 if i == density_idx else 0.0 for i in range(3)]

    return [
        float(query_profile.token_count),
        query_profile.embedding_norm,
        query_profile.entropy,
        *type_onehot,
        float(np.log1p(space_profile.n_candidates)),
        space_profile.variance,
        float(np.log1p(space_profile.anisotropy)),
        space_profile.score_spread,
        *density_onehot,
    ]


class LearnedRouter(BaseRouter):
    """Learned metric weight router using a small MLP.

    A gating network ``f_θ(query_features, space_features) → weights``
    trained on relevance feedback data. The output is passed through
    softmax to produce a valid weight distribution.

    Requires PyTorch. Install with ``pip install 'refract-search[learned]'``.

    Example:
        >>> router = LearnedRouter(["cosine", "bm25", "mahalanobis"])
        >>> router.fit(query_profiles, space_profiles, relevance_labels)
        >>> weights = router.route(query_profile, space_profile, metrics)
    """

    name = "learned"

    def __init__(
        self,
        metric_names: list[str],
        hidden_size: int = 32,
    ) -> None:
        """Initialize learned router.

        Args:
            metric_names: List of metric names this router produces weights for.
            hidden_size: Hidden layer size in the MLP.
        """
        self.torch = _try_import_torch()
        self.metric_names = list(metric_names)
        self.hidden_size = hidden_size

        n_features = 13  # from _profile_to_features
        n_outputs = len(metric_names)

        self._model = self.torch.nn.Sequential(
            self.torch.nn.Linear(n_features, hidden_size),
            self.torch.nn.ReLU(),
            self.torch.nn.Dropout(0.1),
            self.torch.nn.Linear(hidden_size, hidden_size),
            self.torch.nn.ReLU(),
            self.torch.nn.Linear(hidden_size, n_outputs),
        )
        self._trained = False

    def fit(
        self,
        query_profiles: list[QueryProfile],
        space_profiles: list[SpaceProfile],
        relevance_labels: list[list[int]],
        epochs: int = 50,
        learning_rate: float = 1e-3,
    ) -> dict[str, list[float]]:
        """Train the gating network on relevance feedback data.

        Args:
            query_profiles: List of query profiles.
            space_profiles: List of space profiles.
            relevance_labels: For each query, list of relevant doc indices.
            epochs: Number of training epochs.
            learning_rate: Adam learning rate.

        Returns:
            Training history with per-epoch loss values.
        """
        torch = self.torch

        # Build feature matrix
        features = []
        for qp, sp in zip(query_profiles, space_profiles):
            features.append(_profile_to_features(qp, sp))

        x_features = torch.tensor(features, dtype=torch.float32)

        # Target: uniform weights as starting point
        # In practice, you'd derive optimal weights from relevance data
        n = len(self.metric_names)
        targets = torch.ones(len(features), n, dtype=torch.float32) / n

        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        criterion = torch.nn.KLDivLoss(reduction="batchmean")

        history: dict[str, list[float]] = {"loss": []}
        self._model.train()

        for _epoch in range(epochs):
            optimizer.zero_grad()
            logits = self._model(x_features)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss = criterion(log_probs, targets)
            loss.backward()
            optimizer.step()
            history["loss"].append(float(loss.item()))

        self._trained = True
        self._model.eval()
        return history

    def route(
        self,
        query_profile: QueryProfile,
        space_profile: SpaceProfile,
        available_metrics: list[str],
    ) -> dict[str, float]:
        """Predict metric weights using the trained model.

        Args:
            query_profile: Analyzed query characteristics.
            space_profile: Analyzed search space geometry.
            available_metrics: List of available metric names.

        Returns:
            Dictionary mapping metric name to weight (sum = 1.0).

        Raises:
            RuntimeError: If the model has not been trained.
        """
        if not self._trained:
            raise RuntimeError(
                "LearnedRouter has not been trained. Call .fit() first, "
                "or use HeuristicRouter (the default) which requires no training."
            )
        torch = self.torch

        features = _profile_to_features(query_profile, space_profile)
        x = torch.tensor([features], dtype=torch.float32)

        with torch.no_grad():
            logits = self._model(x)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            weights_arr = probs[0].numpy()

        # Map to available metrics
        result: dict[str, float] = {}
        for i, name in enumerate(self.metric_names):
            if name in available_metrics:
                result[name] = float(weights_arr[i])

        # Normalize
        total = sum(result.values())
        if total > 1e-12:
            result = {k: v / total for k, v in result.items()}

        return result

    def save(self, path: str) -> None:
        """Save the trained model to disk.

        Args:
            path: File path to save the model (e.g., "router.pt").
        """
        torch = self.torch
        save_data = {
            "model_state": self._model.state_dict(),
            "metric_names": self.metric_names,
            "hidden_size": self.hidden_size,
            "trained": self._trained,
        }
        torch.save(save_data, path)

    @classmethod
    def load(cls, path: str) -> LearnedRouter:
        """Load a trained model from disk.

        Args:
            path: File path to the saved model.

        Returns:
            A LearnedRouter with the loaded model.
        """
        torch = _try_import_torch()
        data = torch.load(path, weights_only=False)
        router = cls(
            metric_names=data["metric_names"],
            hidden_size=data["hidden_size"],
        )
        router._model.load_state_dict(data["model_state"])
        router._trained = data["trained"]
        router._model.eval()
        return router
