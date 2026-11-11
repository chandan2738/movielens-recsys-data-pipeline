"""Recommender model implementations: Baseline Popularity and Hybrid ALS+Implicit+Content."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


class PopularityRecommender:
    """Non-personalized baseline recommender using Bayesian smoothed popularity."""

    def __init__(self, bayesian_m: float = 25.0, popularity_weight: float = 0.15) -> None:
        self.bayesian_m = bayesian_m
        self.popularity_weight = popularity_weight
        self.global_mean: float = 3.5
        self.item_ids: np.ndarray = np.array([], dtype=np.int64)
        self.item_to_idx: dict[int, int] = {}
        self.item_bayesian: np.ndarray = np.array([], dtype=np.float64)
        self.item_scores: np.ndarray = np.array([], dtype=np.float64)
        self.seen_by_user: dict[int, set[int]] = {}

    def fit(self, item_features: pd.DataFrame, train_df: pd.DataFrame) -> "PopularityRecommender":
        self.global_mean = float(train_df["rating"].mean())
        items_sorted = item_features.sort_values("item_id").reset_index(drop=True)
        self.item_ids = items_sorted["item_id"].to_numpy(dtype=np.int64)
        self.item_to_idx = {int(iid): idx for idx, iid in enumerate(self.item_ids)}

        counts = items_sorted["item_popularity_count"].to_numpy(dtype=np.float64)
        means = items_sorted["item_avg_rating"].to_numpy(dtype=np.float64)
        self.item_bayesian = (counts * means + self.bayesian_m * self.global_mean) / (counts + self.bayesian_m)
        self.item_scores = self.item_bayesian + self.popularity_weight * np.log1p(counts)

        self.seen_by_user = (
            train_df.groupby("user_id")["item_id"]
            .apply(lambda s: set(int(x) for x in s))
            .to_dict()
        )
        return self

    def predict_ratings(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        preds = np.full(len(item_ids), self.global_mean, dtype=np.float64)
        for pos, iid in enumerate(item_ids):
            idx = self.item_to_idx.get(int(iid))
            if idx is not None:
                preds[pos] = self.item_bayesian[idx]
        return np.clip(preds, 1.0, 5.0)

    def recommend_for_user(self, user_id: int, top_k: int = 10, exclude_seen: bool = True) -> list[tuple[int, float]]:
        scores = self.item_scores.copy()
        if exclude_seen:
            for seen_iid in self.seen_by_user.get(int(user_id), ()):
                idx = self.item_to_idx.get(seen_iid)
                if idx is not None:
                    scores[idx] = -np.inf
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [(int(self.item_ids[idx]), float(scores[idx])) for idx in top_indices]


class HybridSVDRecommender:
    """Personalized Hybrid Recommender combining Biased ALS MF, Implicit SVD, and Feature Store genre affinity."""

    def __init__(
        self,
        n_factors: int = 32,
        n_epochs: int = 20,
        reg: float = 10.0,
        bias_reg: float = 5.0,
        implicit_svd_factors: int = 24,
        explicit_weight: float = 0.35,
        implicit_weight: float = 0.50,
        genre_weight: float = 0.15,
        random_state: int = 42,
    ) -> None:
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.reg = reg
        self.bias_reg = bias_reg
        self.implicit_svd_factors = implicit_svd_factors
        self.explicit_weight = explicit_weight
        self.implicit_weight = implicit_weight
        self.genre_weight = genre_weight
        self.random_state = random_state

        self.global_mean: float = 3.5
        self.user_ids: np.ndarray = np.array([], dtype=np.int64)
        self.item_ids: np.ndarray = np.array([], dtype=np.int64)
        self.user_to_idx: dict[int, int] = {}
        self.item_to_idx: dict[int, int] = {}
        self.seen_by_user: dict[int, set[int]] = {}

        self.user_bias: np.ndarray = np.array([], dtype=np.float64)
        self.item_bias: np.ndarray = np.array([], dtype=np.float64)
        self.user_factors: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self.item_factors: np.ndarray = np.empty((0, 0), dtype=np.float64)

        self.implicit_user_factors: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self.implicit_item_factors: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self.genre_affinity_matrix: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self.training_loss_history: list[float] = []

    def fit(
        self,
        rating_matrix: sparse.csr_matrix,
        implicit_matrix: sparse.csr_matrix,
        user_ids: list[int],
        item_ids: list[int],
        user_genre_profile: pd.DataFrame,
        item_genre_vectors: pd.DataFrame,
        genre_columns: list[str],
        train_df: pd.DataFrame,
    ) -> "HybridSVDRecommender":
        rng = np.random.default_rng(self.random_state)
        self.user_ids = np.asarray(user_ids, dtype=np.int64)
        self.item_ids = np.asarray(item_ids, dtype=np.int64)
        self.user_to_idx = {int(uid): idx for idx, uid in enumerate(self.user_ids)}
        self.item_to_idx = {int(iid): idx for idx, iid in enumerate(self.item_ids)}
        self.seen_by_user = (
            train_df.groupby("user_id")["item_id"]
            .apply(lambda s: set(int(x) for x in s))
            .to_dict()
        )

        n_users, n_items = rating_matrix.shape
        self.global_mean = float(rating_matrix.data.mean())

        self.user_bias = np.zeros(n_users, dtype=np.float64)
        self.item_bias = np.zeros(n_items, dtype=np.float64)
        self.user_factors = rng.normal(0.0, 0.1, size=(n_users, self.n_factors))
        self.item_factors = rng.normal(0.0, 0.1, size=(n_items, self.n_factors))

        csr = rating_matrix.tocsr()
        csc = rating_matrix.tocsc()
        eye_f = self.reg * np.eye(self.n_factors, dtype=np.float64)

        self.training_loss_history = []
        for _ in range(self.n_epochs):
            # Update user biases & factors
            for u in range(n_users):
                start, end = csr.indptr[u], csr.indptr[u + 1]
                if start == end:
                    continue
                cols = csr.indices[start:end]
                vals = csr.data[start:end].astype(np.float64)
                q_u = self.item_factors[cols]
                pred_dot = q_u @ self.user_factors[u]
                self.user_bias[u] = np.sum(vals - self.global_mean - self.item_bias[cols] - pred_dot) / (
                    len(cols) + self.bias_reg
                )
                target = vals - self.global_mean - self.user_bias[u] - self.item_bias[cols]
                a_mat = q_u.T @ q_u + eye_f
                b_vec = q_u.T @ target
                self.user_factors[u] = np.linalg.solve(a_mat, b_vec)

            # Update item biases & factors
            for i in range(n_items):
                start, end = csc.indptr[i], csc.indptr[i + 1]
                if start == end:
                    continue
                rows = csc.indices[start:end]
                vals = csc.data[start:end].astype(np.float64)
                p_i = self.user_factors[rows]
                pred_dot = p_i @ self.item_factors[i]
                self.item_bias[i] = np.sum(vals - self.global_mean - self.user_bias[rows] - pred_dot) / (
                    len(rows) + self.bias_reg
                )
                target = vals - self.global_mean - self.user_bias[rows] - self.item_bias[i]
                a_mat = p_i.T @ p_i + eye_f
                b_vec = p_i.T @ target
                self.item_factors[i] = np.linalg.solve(a_mat, b_vec)

            # Compute train RMSE checkpoint
            coo = csr.tocoo()
            preds = (
                self.global_mean
                + self.user_bias[coo.row]
                + self.item_bias[coo.col]
                + np.sum(self.user_factors[coo.row] * self.item_factors[coo.col], axis=1)
            )
            rmse = float(np.sqrt(np.mean((coo.data - preds) ** 2)))
            self.training_loss_history.append(round(rmse, 5))

        # Implicit collaborative filtering via TruncatedSVD on L2-normalized positive interaction matrix
        imp_norm = normalize(implicit_matrix.astype(np.float64), norm="l2", axis=1)
        svd = TruncatedSVD(n_components=self.implicit_svd_factors, random_state=self.random_state)
        self.implicit_user_factors = svd.fit_transform(imp_norm)
        self.implicit_item_factors = svd.components_.T

        # Content genre similarity matrix from Feature Store profiles
        ug = user_genre_profile.set_index("user_id").loc[self.user_ids, genre_columns].to_numpy(dtype=np.float64)
        ig = item_genre_vectors.set_index("item_id").loc[self.item_ids, genre_columns].to_numpy(dtype=np.float64)
        ug_norm = ug / (np.linalg.norm(ug, axis=1, keepdims=True) + 1e-12)
        ig_norm = ig / (np.linalg.norm(ig, axis=1, keepdims=True) + 1e-12)
        self.genre_affinity_matrix = ug_norm @ ig_norm.T
        return self

    def predict_ratings(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        preds = np.full(len(user_ids), self.global_mean, dtype=np.float64)
        for pos, (uid, iid) in enumerate(zip(user_ids, item_ids)):
            u_idx = self.user_to_idx.get(int(uid))
            i_idx = self.item_to_idx.get(int(iid))
            if u_idx is not None and i_idx is not None:
                preds[pos] = (
                    self.global_mean
                    + self.user_bias[u_idx]
                    + self.item_bias[i_idx]
                    + float(self.user_factors[u_idx] @ self.item_factors[i_idx])
                )
            elif u_idx is not None:
                preds[pos] = self.global_mean + self.user_bias[u_idx]
            elif i_idx is not None:
                preds[pos] = self.global_mean + self.item_bias[i_idx]
        return np.clip(preds, 1.0, 5.0)

    @staticmethod
    def _minmax(arr: np.ndarray) -> np.ndarray:
        min_v = float(np.min(arr))
        max_v = float(np.max(arr))
        if max_v - min_v < 1e-12:
            return np.zeros_like(arr)
        return (arr - min_v) / (max_v - min_v)

    def recommend_for_user(self, user_id: int, top_k: int = 10, exclude_seen: bool = True) -> list[tuple[int, float]]:
        u_idx = self.user_to_idx.get(int(user_id))
        if u_idx is None:
            return []

        explicit_scores = np.clip(
            self.global_mean
            + self.user_bias[u_idx]
            + self.item_bias
            + (self.item_factors @ self.user_factors[u_idx]),
            1.0,
            5.0,
        )
        implicit_scores = self.implicit_item_factors @ self.implicit_user_factors[u_idx]
        genre_scores = self.genre_affinity_matrix[u_idx]

        blended = (
            self.explicit_weight * self._minmax(explicit_scores)
            + self.implicit_weight * self._minmax(implicit_scores)
            + self.genre_weight * self._minmax(genre_scores)
        )

        if exclude_seen:
            for seen_iid in self.seen_by_user.get(int(user_id), ()):
                idx = self.item_to_idx.get(seen_iid)
                if idx is not None:
                    blended[idx] = -np.inf

        top_indices = np.argpartition(blended, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(blended[top_indices])[::-1]]
        return [(int(self.item_ids[idx]), float(blended[idx])) for idx in top_indices]

    def get_params(self) -> dict[str, Any]:
        return {
            "n_factors": self.n_factors,
            "n_epochs": self.n_epochs,
            "reg": self.reg,
            "bias_reg": self.bias_reg,
            "implicit_svd_factors": self.implicit_svd_factors,
            "explicit_weight": self.explicit_weight,
            "implicit_weight": self.implicit_weight,
            "genre_weight": self.genre_weight,
            "final_train_rmse": self.training_loss_history[-1] if self.training_loss_history else None,
        }

