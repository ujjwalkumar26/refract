"""Learned router — train metric routing from relevance feedback."""

from __future__ import annotations

import dataclasses
import pickle
from typing import TYPE_CHECKING

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from refract.analysis.query_analyzer import analyze_query
from refract.analysis.space_analyzer import analyze_space
from refract.benchmark.eval_metrics import ndcg_at_k
from refract.metrics.bm25 import BM25Metric
from refract.metrics.cosine import CosineMetric
from refract.routing.base import BaseRouter
from refract.search import _build_tfidf_vectors, _resolve_metrics

if TYPE_CHECKING:
    from collections.abc import Sequence

    from refract.embedders.base import BaseEmbedder
    from refract.metrics.base import BaseMetric
    from refract.types import QueryProfile, SpaceProfile


def _profile_to_features(query_profile: QueryProfile, space_profile: SpaceProfile) -> list[float]:
    """Convert profiles into a stable feature vector."""
    type_map = {"keyword": 0, "natural_language": 1, "code": 2, "structured": 3}
    type_idx = type_map.get(query_profile.query_type, 1)
    type_onehot = [1.0 if i == type_idx else 0.0 for i in range(4)]

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


def _normalize_weights(weights: Sequence[float]) -> np.ndarray:
    """Project a raw vector onto the probability simplex."""
    arr = np.asarray(weights, dtype=np.float64)
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total < 1e-12:
        return np.asarray(np.ones_like(arr) / max(len(arr), 1), dtype=np.float64)
    return np.asarray(arr / total, dtype=np.float64)


@dataclasses.dataclass(frozen=True)
class LearnedRouterEvaluation:
    """Summary of learned-router quality on a labeled dataset."""

    n_queries: int
    top_k: int
    weight_mae: float
    router_ndcg_at_k: float
    oracle_ndcg_at_k: float
    metric_quality: dict[str, float]

    def __repr__(self) -> str:
        return (
            "LearnedRouterEvaluation("
            f"n_queries={self.n_queries}, "
            f"weight_mae={self.weight_mae:.3f}, "
            f"router_ndcg@{self.top_k}={self.router_ndcg_at_k:.3f}, "
            f"oracle_ndcg@{self.top_k}={self.oracle_ndcg_at_k:.3f})"
        )


class LearnedRouter(BaseRouter):
    """Trainable router that predicts metric weights from query/space features.

    The router learns against per-query target weight distributions. In the
    common case you do not need to construct those targets yourself:
    ``fit_from_relevance()`` derives them directly from relevance judgments by
    measuring how well each metric ranks the labeled documents.
    """

    name = "learned"

    def __init__(
        self,
        metric_names: list[str],
        hidden_size: int = 24,
        random_state: int = 42,
    ) -> None:
        if not metric_names:
            raise ValueError("LearnedRouter requires at least one metric name.")

        self.metric_names = list(metric_names)
        self.hidden_size = hidden_size
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._model = MLPRegressor(
            hidden_layer_sizes=(hidden_size, hidden_size),
            activation="relu",
            solver="lbfgs",
            random_state=random_state,
            max_iter=500,
        )
        self._trained = False

    def fit(
        self,
        query_profiles: list[QueryProfile],
        space_profiles: list[SpaceProfile],
        target_weights: list[dict[str, float]],
    ) -> LearnedRouter:
        """Fit the router from explicit target weight distributions."""
        if not query_profiles or not space_profiles or not target_weights:
            raise ValueError(
                "fit() requires non-empty query_profiles, space_profiles, and target_weights."
            )
        if not (len(query_profiles) == len(space_profiles) == len(target_weights)):
            raise ValueError("fit() inputs must have the same length.")

        feature_rows = [
            _profile_to_features(qp, sp) for qp, sp in zip(query_profiles, space_profiles)
        ]
        targets = np.vstack(
            [
                _normalize_weights([target.get(metric, 0.0) for metric in self.metric_names])
                for target in target_weights
            ]
        )

        scaled = self._scaler.fit_transform(np.asarray(feature_rows, dtype=np.float64))
        self._model.fit(scaled, targets)
        self._trained = True
        return self

    def fit_from_relevance(
        self,
        queries: list[str] | np.ndarray,
        corpus: Sequence[str] | np.ndarray,
        relevance: dict[int, set[int]],
        *,
        embedder: BaseEmbedder | None = None,
        top_k: int = 10,
        target_temperature: float = 0.15,
    ) -> LearnedRouterEvaluation:
        """Fit the router directly from relevance judgments."""
        query_ids, query_profiles, space_profiles, targets, metric_quality, oracle_ndcg = (
            self._derive_training_targets(
                queries=queries,
                corpus=corpus,
                relevance=relevance,
                embedder=embedder,
                top_k=top_k,
                target_temperature=target_temperature,
            )
        )
        self.fit(query_profiles, space_profiles, targets)
        return self.evaluate_from_relevance(
            queries=queries,
            corpus=corpus,
            relevance=relevance,
            embedder=embedder,
            top_k=top_k,
            query_ids=query_ids,
            expected_targets=targets,
            metric_quality=metric_quality,
            oracle_ndcg_at_k=oracle_ndcg,
        )

    def evaluate_from_relevance(
        self,
        queries: list[str] | np.ndarray,
        corpus: Sequence[str] | np.ndarray,
        relevance: dict[int, set[int]],
        *,
        embedder: BaseEmbedder | None = None,
        top_k: int = 10,
        query_ids: list[int] | None = None,
        expected_targets: list[dict[str, float]] | None = None,
        metric_quality: dict[str, float] | None = None,
        oracle_ndcg_at_k: float | None = None,
    ) -> LearnedRouterEvaluation:
        """Evaluate a trained router on labeled data."""
        if not self._trained:
            raise RuntimeError(
                "LearnedRouter has not been trained. Call .fit() or .fit_from_relevance() first."
            )

        _, _, all_query_profiles, all_space_profiles, metric_scores = (
            self._prepare_relevance_problem(
                queries=queries,
                corpus=corpus,
                embedder=embedder,
            )
        )

        if expected_targets is None or metric_quality is None or oracle_ndcg_at_k is None:
            (
                query_ids,
                query_profiles,
                space_profiles,
                expected_targets,
                metric_quality,
                oracle_ndcg_at_k,
            ) = self._derive_training_targets(
                queries=queries,
                corpus=corpus,
                relevance=relevance,
                embedder=embedder,
                top_k=top_k,
            )
        else:
            query_ids = query_ids or sorted(idx for idx, docs in relevance.items() if docs)
            query_profiles = [all_query_profiles[idx] for idx in query_ids]
            space_profiles = [all_space_profiles[idx] for idx in query_ids]

        errors: list[float] = []
        router_ndcgs: list[float] = []

        for query_id, query_profile, space_profile, target in zip(
            query_ids, query_profiles, space_profiles, expected_targets
        ):
            relevant = relevance.get(query_id, set())
            if not relevant:
                continue

            predicted = self.route(query_profile, space_profile, self.metric_names)
            predicted_arr = np.array([predicted.get(metric, 0.0) for metric in self.metric_names])
            target_arr = np.array([target.get(metric, 0.0) for metric in self.metric_names])
            errors.append(float(np.abs(predicted_arr - target_arr).mean()))

            weighted_scores = np.zeros_like(metric_scores[self.metric_names[0]][query_id])
            for metric_name, scores in metric_scores.items():
                weighted_scores = weighted_scores + (
                    scores[query_id] * predicted.get(metric_name, 0.0)
                )

            ranking = np.argsort(-weighted_scores)[:top_k].tolist()
            router_ndcgs.append(ndcg_at_k(ranking, relevant, top_k))

        n_queries = len(errors)
        return LearnedRouterEvaluation(
            n_queries=n_queries,
            top_k=top_k,
            weight_mae=float(np.mean(errors)) if errors else 0.0,
            router_ndcg_at_k=float(np.mean(router_ndcgs)) if router_ndcgs else 0.0,
            oracle_ndcg_at_k=oracle_ndcg_at_k,
            metric_quality=metric_quality,
        )

    def route(
        self,
        query_profile: QueryProfile,
        space_profile: SpaceProfile,
        available_metrics: list[str],
    ) -> dict[str, float]:
        if not self._trained:
            raise RuntimeError(
                "LearnedRouter has not been trained. Call .fit() first, "
                "or use HeuristicRouter when you do not have relevance data."
            )

        features = np.asarray(
            [_profile_to_features(query_profile, space_profile)], dtype=np.float64
        )
        scaled = self._scaler.transform(features)
        prediction = self._model.predict(scaled)[0]
        normalized = _normalize_weights(prediction)

        weights = {
            metric_name: float(normalized[idx])
            for idx, metric_name in enumerate(self.metric_names)
            if metric_name in available_metrics
        }
        total = sum(weights.values())
        if total < 1e-12:
            n = len(available_metrics)
            return {metric: 1.0 / n for metric in available_metrics} if n > 0 else {}
        return {metric: value / total for metric, value in weights.items()}

    def save(self, path: str) -> None:
        """Persist the trained router to disk."""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "metric_names": self.metric_names,
                    "hidden_size": self.hidden_size,
                    "random_state": self.random_state,
                    "scaler": self._scaler,
                    "model": self._model,
                    "trained": self._trained,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> LearnedRouter:
        """Load a trained router from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        router = cls(
            metric_names=list(data["metric_names"]),
            hidden_size=int(data["hidden_size"]),
            random_state=int(data["random_state"]),
        )
        router._scaler = data["scaler"]
        router._model = data["model"]
        router._trained = bool(data["trained"])
        return router

    def _derive_training_targets(
        self,
        *,
        queries: list[str] | np.ndarray,
        corpus: Sequence[str] | np.ndarray,
        relevance: dict[int, set[int]],
        embedder: BaseEmbedder | None = None,
        top_k: int = 10,
        target_temperature: float = 0.15,
    ) -> tuple[
        list[int],
        list[QueryProfile],
        list[SpaceProfile],
        list[dict[str, float]],
        dict[str, float],
        float,
    ]:
        (
            _query_texts,
            _corpus_texts,
            query_profiles,
            space_profiles,
            metric_scores,
        ) = self._prepare_relevance_problem(
            queries=queries,
            corpus=corpus,
            embedder=embedder,
        )

        labeled_ids: list[int] = []
        labeled_profiles: list[QueryProfile] = []
        labeled_spaces: list[SpaceProfile] = []
        targets: list[dict[str, float]] = []
        oracle_ndcgs: list[float] = []
        quality_by_metric: dict[str, list[float]] = {name: [] for name in self.metric_names}

        for idx, query_profile in enumerate(query_profiles):
            relevant = relevance.get(idx, set())
            if not relevant:
                continue
            labeled_ids.append(idx)
            labeled_profiles.append(query_profile)
            labeled_spaces.append(space_profiles[idx])

            utilities: list[float] = []
            for metric_name in self.metric_names:
                scores = metric_scores[metric_name][idx]
                ranking = np.argsort(-scores)[:top_k].tolist()
                utility = ndcg_at_k(ranking, relevant, top_k)
                utilities.append(utility)
                quality_by_metric[metric_name].append(utility)

            targets_arr = self._utilities_to_targets(utilities, temperature=target_temperature)
            target = {
                metric_name: float(targets_arr[m_idx])
                for m_idx, metric_name in enumerate(self.metric_names)
            }
            targets.append(target)

            oracle_scores = np.zeros_like(metric_scores[self.metric_names[0]][idx])
            for metric_name, weight in target.items():
                oracle_scores = oracle_scores + (metric_scores[metric_name][idx] * weight)
            oracle_ranking = np.argsort(-oracle_scores)[:top_k].tolist()
            oracle_ndcgs.append(ndcg_at_k(oracle_ranking, relevant, top_k))

        metric_quality = {
            metric_name: float(np.mean(values)) if values else 0.0
            for metric_name, values in quality_by_metric.items()
        }
        oracle_ndcg = float(np.mean(oracle_ndcgs)) if oracle_ndcgs else 0.0
        return labeled_ids, labeled_profiles, labeled_spaces, targets, metric_quality, oracle_ndcg

    def _prepare_relevance_problem(
        self,
        *,
        queries: list[str] | np.ndarray,
        corpus: Sequence[str] | np.ndarray,
        embedder: BaseEmbedder | None = None,
    ) -> tuple[
        list[str] | None,
        list[str] | None,
        list[QueryProfile],
        list[SpaceProfile],
        dict[str, np.ndarray],
    ]:
        query_texts: list[str] | None = None
        query_vecs: np.ndarray
        corpus_texts: list[str] | None = None
        corpus_vecs: np.ndarray

        if isinstance(corpus, np.ndarray):
            corpus_vecs = corpus.astype(np.float64)
        else:
            corpus_texts = list(corpus)
            if embedder is not None:
                corpus_vecs = embedder.embed(corpus_texts).astype(np.float64)
            else:
                if isinstance(queries, np.ndarray):
                    raise ValueError(
                        "Vector queries cannot be trained against a text corpus without an embedder."
                    )
                else:
                    all_texts = corpus_texts + list(queries)
                    all_vecs = _build_tfidf_vectors(all_texts)
                    corpus_vecs = all_vecs[: len(corpus_texts)]
                    query_vecs = all_vecs[len(corpus_texts) :]

        if isinstance(queries, np.ndarray):
            query_vecs = queries.astype(np.float64)
        else:
            query_texts = list(queries)
            if embedder is not None:
                query_vecs = embedder.embed(query_texts).astype(np.float64)
            elif corpus_texts is None:
                raise ValueError(
                    "Text queries cannot be trained against a vector-only corpus without an embedder."
                )

        metric_instances = _resolve_metrics(
            metrics_arg=[*self.metric_names],
            corpus_texts=corpus_texts,
            corpus_vecs=corpus_vecs,
        )
        missing_metrics = [name for name in self.metric_names if name not in metric_instances]
        if missing_metrics:
            raise ValueError(
                f"Unable to prepare metrics for {missing_metrics}. "
                "Text metrics like BM25 require a text corpus."
            )

        query_profiles: list[QueryProfile] = []
        space_profiles: list[SpaceProfile] = []
        metric_scores: dict[str, list[np.ndarray]] = {name: [] for name in self.metric_names}
        cosine_metric = CosineMetric()

        for idx in range(len(query_vecs)):
            query_text = query_texts[idx] if query_texts is not None else None
            query_vec = query_vecs[idx]
            cosine_scores = cosine_metric.batch_score(query_vec, corpus_vecs)

            query_profile = analyze_query(
                query_text=query_text,
                query_vector=query_vec,
                candidate_vectors=corpus_vecs,
                candidate_scores=cosine_scores,
            )
            space_profile = analyze_space(corpus_vecs, cosine_scores)
            query_profiles.append(query_profile)
            space_profiles.append(space_profile)

            for metric_name, metric in metric_instances.items():
                metric_scores[metric_name].append(
                    self._score_metric(
                        metric=metric,
                        query_text=query_text,
                        query_vector=query_vec,
                        corpus_vectors=corpus_vecs,
                    )
                )

        stacked_scores = {
            metric_name: np.stack(score_list, axis=0)
            for metric_name, score_list in metric_scores.items()
        }
        return query_texts, corpus_texts, query_profiles, space_profiles, stacked_scores

    def _score_metric(
        self,
        *,
        metric: BaseMetric,
        query_text: str | None,
        query_vector: np.ndarray,
        corpus_vectors: np.ndarray,
    ) -> np.ndarray:
        if metric.is_text_metric and isinstance(metric, BM25Metric):
            if query_text is None:
                return np.zeros(len(corpus_vectors), dtype=np.float64)
            return metric.batch_score_text(query_text)
        return metric.batch_score(query_vector, corpus_vectors)

    def _utilities_to_targets(
        self,
        utilities: Sequence[float],
        *,
        temperature: float,
    ) -> np.ndarray:
        raw = np.asarray(utilities, dtype=np.float64)
        if np.all(raw <= 1e-12):
            return np.asarray(
                np.ones(len(raw), dtype=np.float64) / max(len(raw), 1), dtype=np.float64
            )

        scaled = raw / max(temperature, 1e-6)
        scaled = scaled - float(np.max(scaled))
        exp_values = np.exp(scaled)
        return _normalize_weights(exp_values)
