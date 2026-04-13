import numpy as np
import pandas as pd
from pathlib import Path

MVO_LAMBDA = 0.5
MVO_RIDGE = 1e-3
WEIGHT_CAP = 0.05

LINEAR_ASSETS  = ["asset_1",  "asset_2",  "asset_3"]
MIXED_ASSETS   = ["asset_4",  "asset_5",  "asset_6"]
SPECIAL_ASSETS = ["asset_7",  "asset_8",  "asset_9",  "asset_10"]
NOISE_ASSETS   = ["asset_11", "asset_12"]


class Polynomial:
    def __init__(self) -> None:
        self._pairs: list = []
        self.n_features_out_: int = 0

    def fit(self, X: np.ndarray) -> "Polynomial":
        d = X.shape[1]
        self._pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
        self.n_features_out_ = d + d + len(self._pairs)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        squared = X ** 2
        if self._pairs:
            cross = np.column_stack([X[:, i] * X[:, j] for i, j in self._pairs])
        else:
            cross = np.empty((len(X), 0))
        return np.concatenate([X, squared, cross], axis=1)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class PolynomialRidge:
    def __init__(self, lambda_reg: float = 10.0) -> None:
        self.lambda_reg = lambda_reg
        self.W: np.ndarray = None

    def fit(self, X_poly_n: np.ndarray, y: np.ndarray) -> None:
        d = X_poly_n.shape[1]
        A = X_poly_n.T @ X_poly_n + self.lambda_reg * np.eye(d)
        self.W = np.linalg.solve(A, X_poly_n.T @ y)

    def predict(self, X_poly_n: np.ndarray) -> np.ndarray:
        return X_poly_n @ self.W


class SubmissionScorerTemplate:
    def __init__(self):
        self.asset_cols = []
        self.pred_cols = []
        self.weight_cols = []

    def set_assets(self, asset_cols):
        self.asset_cols = list(asset_cols)
        self.pred_cols = [f"pred_{col}" for col in self.asset_cols]
        self.weight_cols = [f"weight_{col}" for col in self.asset_cols]

    def validate_submission_format(self, submission):
        required = ["id", "date"] + self.pred_cols + self.weight_cols
        missing = [col for col in required if col not in submission.columns]
        if missing:
            raise ValueError(f"Missing columns in submission: {missing}")
        if submission[required].isnull().any().any():
            raise ValueError("Submission contains missing values.")
        if submission["id"].duplicated().any():
            raise ValueError("Submission contains duplicate ids.")

    def score_predictions(self, predictions, realized_returns):
        common_assets = [col for col in self.asset_cols if col in realized_returns.columns]
        if not common_assets:
            raise ValueError("No common asset columns found for scoring.")
        pred_cols = [f"pred_{col}" for col in common_assets]
        merged = realized_returns[["id", "date"] + common_assets].merge(
            predictions[["id", "date"] + pred_cols], on=["id", "date"], how="inner"
        )
        if merged.empty:
            raise ValueError("No overlapping rows found between predictions and realized returns.")
        y_true = merged[common_assets].to_numpy(dtype=float)
        y_pred = merged[pred_cols].to_numpy(dtype=float)
        return {"mse": float(np.mean((y_true - y_pred) ** 2)), "n_rows": int(len(merged))}


class PortfolioChallengeModel:
    def __init__(self):
        self.feature_cols = []
        self.asset_cols = []
        self.pred_cols = []
        self.weight_cols = []

    def read_table(self, path):
        path = Path(path)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported file format: {path}. Expected .parquet")

    def load_data(self, data_dir="data"):
        data_dir = Path(data_dir)
        x_train = self.read_table(data_dir / "X_train.parquet")
        r_train = self.read_table(data_dir / "R_train.parquet")
        x_test = self.read_table(data_dir / "X_test.parquet")

        self.feature_cols = [col for col in x_train.columns if col not in ["id", "date", "split"]]
        self.asset_cols = [col for col in r_train.columns if col not in ["id", "date", "split"]]
        self.pred_cols = [f"pred_{col}" for col in self.asset_cols]
        self.weight_cols = [f"weight_{col}" for col in self.asset_cols]

        return {"X_train": x_train, "R_train": r_train, "X_test": x_test}

    def fit(self, x_train, r_train,
            lambda_linear=1.0, lambda_special=1.0,
            poly_lambda=None,
            verbose=True):
        return self._fit(x_train, r_train,
                         lambda_linear=lambda_linear,
                         lambda_special=lambda_special,
                         poly_lambda=poly_lambda,
                         verbose=verbose)

    def _fit(self, x_train, r_train,
                   lambda_linear=1.0, lambda_special=1.0,
                   poly_lambda=None,
                   verbose=True):
        X = x_train[self.feature_cols].to_numpy(dtype=float)
        y = r_train[self.asset_cols].to_numpy(dtype=float)

        n = X.shape[0]
        split = max(1, int(0.9 * n))
        X_tr, y_tr = X[:split], y[:split]
        X_va, y_va = X[split:], y[split:]

        self.x_mean = X_tr.mean(axis=0, keepdims=True)
        self.x_std  = X_tr.std(axis=0, keepdims=True) + 1e-8
        X_tr_n = (X_tr - self.x_mean) / self.x_std
        X_va_n = (X_va - self.x_mean) / self.x_std

        self.y_mean = y_tr.mean(axis=0, keepdims=True)
        self.y_std  = y_tr.std(axis=0, keepdims=True) + 1e-8
        y_tr_n = np.clip((y_tr - self.y_mean) / self.y_std, -5.0, 5.0)

        def _indices(names):
            return [i for i, col in enumerate(self.asset_cols) if col in names]

        self.linear_indices  = _indices(LINEAR_ASSETS)
        self.mixed_indices   = _indices(MIXED_ASSETS)
        self.special_indices = _indices(SPECIAL_ASSETS)
        self.noise_indices   = _indices(NOISE_ASSETS)

        d = X_tr_n.shape[1]
        A = X_tr_n.T @ X_tr_n + lambda_linear * np.eye(d)
        self.W_linear = np.linalg.solve(A, X_tr_n.T @ y_tr_n[:, self.linear_indices])

        if verbose:
            pred_va_lin = X_va_n @ self.W_linear
            y_va_lin_n  = (y_va[:, self.linear_indices] - self.y_mean[:, self.linear_indices]) / self.y_std[:, self.linear_indices]
            mse_lin = np.mean((pred_va_lin - y_va_lin_n) ** 2)
            print(f"[Linear   ] closed-form Ridge done — val MSE (norm): {mse_lin:.6f}")

        self.poly = Polynomial()
        X_tr_poly   = self.poly.fit_transform(X_tr_n)
        X_va_poly   = self.poly.transform(X_va_n)

        self.poly_mean = X_tr_poly.mean(axis=0, keepdims=True)
        self.poly_std  = X_tr_poly.std(axis=0, keepdims=True) + 1e-8
        X_tr_poly_n = np.clip((X_tr_poly - self.poly_mean) / self.poly_std, -5.0, 5.0)
        X_va_poly_n = np.clip((X_va_poly - self.poly_mean) / self.poly_std, -5.0, 5.0)

        y_tr_mixed = y_tr_n[:, self.mixed_indices]
        y_va_mix_n = np.clip(
            (y_va[:, self.mixed_indices] - self.y_mean[:, self.mixed_indices]) / self.y_std[:, self.mixed_indices],
            -5.0, 5.0,
        )

        d_poly = X_tr_poly_n.shape[1]
        if poly_lambda is not None:
            best_lambda_mix = poly_lambda
        else:
            best_lambda_mix, best_mse_mix = 1.0, np.inf
            for lam in [0.1, 1.0, 10.0, 100.0, 1000.0, 5000.0, 10000.0]:
                _m = PolynomialRidge(lambda_reg=lam)
                _m.fit(X_tr_poly_n, y_tr_mixed)
                _mse = float(np.mean((_m.predict(X_va_poly_n) - y_va_mix_n) ** 2))
                if _mse < best_mse_mix:
                    best_mse_mix, best_lambda_mix = _mse, lam

        self.mixed_model = PolynomialRidge(lambda_reg=best_lambda_mix)
        self.mixed_model.fit(X_tr_poly_n, y_tr_mixed)

        if verbose:
            pred_va_mix = self.mixed_model.predict(X_va_poly_n)
            mse_mix = float(np.mean((pred_va_mix - y_va_mix_n) ** 2))
            print(f"[Mixed    ] poly Ridge done (dim={d_poly}, λ={best_lambda_mix}) — val MSE (norm): {mse_mix:.6f}")

        y_tr_special = y_tr_n[:, self.special_indices]
        y_va_spe_n   = (y_va[:, self.special_indices] - self.y_mean[:, self.special_indices]) / self.y_std[:, self.special_indices]

        d = X_tr_n.shape[1]
        A_spe = X_tr_n.T @ X_tr_n + lambda_special * np.eye(d)
        self.W_special = np.linalg.solve(A_spe, X_tr_n.T @ y_tr_special)

        if verbose:
            pred_va_spe = X_va_n @ self.W_special
            mse_spe = np.mean((pred_va_spe - y_va_spe_n) ** 2)
            print(f"[Special  ] Ridge done (feat dim={d}) — val MSE (norm): {mse_spe:.6f}")

        Sigma = np.cov(y_tr.T) + MVO_RIDGE * np.eye(len(self.asset_cols))
        self.Sigma_inv = np.linalg.inv(Sigma)

        def _sharpe(r: np.ndarray) -> float:
            mu = r.mean()
            sig = r.std() + 1e-10
            return float(mu / sig * np.sqrt(24 * 252))

        def _mse(pred_n, y_n):
            return float(np.mean((pred_n - y_n) ** 2))

        lin_tr_n = X_tr_n @ self.W_linear
        lin_va_n = X_va_n @ self.W_linear
        mix_tr_n = self.mixed_model.predict(X_tr_poly_n)
        mix_va_n = self.mixed_model.predict(X_va_poly_n)
        spe_tr_n = X_tr_n @ self.W_special
        spe_va_n = X_va_n @ self.W_special

        y_tr_lin_n = y_tr_n[:, self.linear_indices]
        y_tr_spe_n = y_tr_n[:, self.special_indices]
        y_va_lin_n = (y_va[:, self.linear_indices] - self.y_mean[:, self.linear_indices]) / self.y_std[:, self.linear_indices]

        def _portfolio_sharpe(pred_returns: np.ndarray, y_raw: np.ndarray) -> float:
            weights = (MVO_LAMBDA * (self.Sigma_inv @ pred_returns.T)).T
            weights[:, self.noise_indices] = 0.0
            weights = np.clip(weights, -WEIGHT_CAP, WEIGHT_CAP)
            gross = np.abs(weights).sum(axis=1, keepdims=True)
            weights = weights / np.where(gross > 1e-8, gross, 1.0)
            port_ret = (weights * y_raw).sum(axis=1)
            return _sharpe(port_ret)

        def _full_pred_n(lin, mix, spe, n):
            p = np.zeros((n, len(self.asset_cols)))
            p[:, self.linear_indices]  = lin
            p[:, self.mixed_indices]   = mix
            p[:, self.special_indices] = spe
            return p

        full_pred_tr = _full_pred_n(lin_tr_n, mix_tr_n, spe_tr_n, len(X_tr)) * self.y_std + self.y_mean
        full_pred_va = _full_pred_n(lin_va_n, mix_va_n, spe_va_n, len(X_va)) * self.y_std + self.y_mean

        self._fit_history = {
            "train": {
                "mse_linear": _mse(lin_tr_n, y_tr_lin_n),
                "mse_mixed": _mse(mix_tr_n, y_tr_mixed),
                "mse_special": _mse(spe_tr_n, y_tr_spe_n),
                "sharpe": _portfolio_sharpe(full_pred_tr, y_tr),
            },
            "val": {
                "mse_linear": _mse(lin_va_n, y_va_lin_n),
                "mse_mixed": _mse(mix_va_n, y_va_mix_n),
                "mse_special": _mse(spe_va_n, y_va_spe_n),
                "sharpe": _portfolio_sharpe(full_pred_va, y_va),
            },
            "_val_pred": full_pred_va,
            "_val_returns": y_va,
            "_train_pred": full_pred_tr,
            "_train_returns": y_tr,
        }

        if verbose:
            h = self._fit_history
            print(f"\n{'─'*60}")
            print(f"{'Group':<12} {'Train MSE':>10} {'Val MSE':>10}")
            print(f"{'─'*60}")
            print(f"{'Linear':<12} {h['train']['mse_linear']:>10.6f} {h['val']['mse_linear']:>10.6f}")
            print(f"{'Mixed':<12} {h['train']['mse_mixed']:>10.6f} {h['val']['mse_mixed']:>10.6f}")
            print(f"{'Special':<12} {h['train']['mse_special']:>10.6f} {h['val']['mse_special']:>10.6f}")
            print(f"{'─'*60}")
            print(f"Portfolio Sharpe (ann.)  train={h['train']['sharpe']:+.4f}  val={h['val']['sharpe']:+.4f}")
            print(f"{'─'*60}\n")

    def predict_returns(self, x_df):
        n = len(x_df)
        X = x_df[self.feature_cols].to_numpy(dtype=float)
        X_n = (X - self.x_mean) / self.x_std

        pred_norm = np.zeros((n, len(self.asset_cols)))

        pred_norm[:, self.linear_indices] = X_n @ self.W_linear

        X_poly   = self.poly.transform(X_n)
        X_poly_n = np.clip((X_poly - self.poly_mean) / self.poly_std, -5.0, 5.0)
        pred_norm[:, self.mixed_indices] = self.mixed_model.predict(X_poly_n)

        pred_norm[:, self.special_indices] = X_n @ self.W_special

        predictions = pred_norm * self.y_std + self.y_mean
        return pd.DataFrame(predictions, columns=self.pred_cols, index=x_df.index)

    def build_weights(self, pred_df):
        mu = pred_df.to_numpy(dtype=float)
        weights = (MVO_LAMBDA * (self.Sigma_inv @ mu.T)).T

        for idx in self.noise_indices:
            weights[:, idx] = 0.0

        weights = np.clip(weights, -WEIGHT_CAP, WEIGHT_CAP)
        gross = np.abs(weights).sum(axis=1, keepdims=True)
        weights = weights / np.where(gross > 1e-8, gross, 1.0)

        return pd.DataFrame(weights, columns=self.weight_cols, index=pred_df.index)

    def build_submission(self, x_train, x_test):
        x_all = pd.concat([x_train, x_test], axis=0).reset_index(drop=True)
        pred_all = self.predict_returns(x_all).reset_index(drop=True)
        weights_all = self.build_weights(pred_all).reset_index(drop=True)
        index_all = x_all[["id", "date"]].copy().reset_index(drop=True)
        return pd.concat([index_all, pred_all, weights_all], axis=1)

    def save_submission(self, submission, path="submission.parquet"):
        path = Path(__file__).parent / "submission.parquet"
        if path.exists():
            path.unlink()
        submission.to_parquet(path, index=False)
        print(f"Saved submission to {path.resolve()} ({len(submission)} rows)")
        return path

    def fit_predict_save(self, data_dir="data", output_path="submission.parquet"):
        data = self.load_data(data_dir)
        self.fit(data["X_train"], data["R_train"])
        submission = self.build_submission(data["X_train"], data["X_test"])
        self.save_submission(submission, output_path)
        return submission

    def create_scorer(self):
        scorer = SubmissionScorerTemplate()
        scorer.set_assets(self.asset_cols)
        return scorer


def _portfolio_sharpe_from_preds(
    pred_returns: np.ndarray,
    y_raw: np.ndarray,
    sigma_inv: np.ndarray,
    noise_indices: list,
    mvo_lambda: float,
    weight_cap: float,
) -> float:
    weights = (mvo_lambda * (sigma_inv @ pred_returns.T)).T
    weights[:, noise_indices] = 0.0
    weights = np.clip(weights, -weight_cap, weight_cap)
    gross = np.abs(weights).sum(axis=1, keepdims=True)
    weights = weights / np.where(gross > 1e-8, gross, 1.0)
    port_ret = (weights * y_raw).sum(axis=1)
    mu  = port_ret.mean()
    sig = port_ret.std() + 1e-10
    return float(mu / sig * np.sqrt(24 * 252))


def grid_search_portfolio(data_dir: str = "data") -> dict:
    from itertools import product

    lambda_linear_grid = [0.1,  10.0, 100.0]
    lambda_special_grid = [10.0, 100.0, 500.0, 1000.0]
    poly_lambda_grid = [100.0, 1000.0, 5000.0, 10000.0]
    mvo_lambda_grid = [0.1, 0.5, 1.0, 2.0, 5.0]
    weight_cap_grid = [0.02, 0.03, 0.05, 0.10]

    model_combos = list(product(lambda_linear_grid, lambda_special_grid, poly_lambda_grid))
    port_combos  = list(product(mvo_lambda_grid, weight_cap_grid))
    total = len(model_combos) * len(port_combos)

    print(f"Grid search: {len(model_combos)} model fits × "
          f"{len(port_combos)} portfolio combos = {total} evaluations\n")

    best_sharpe = -np.inf
    best_params: dict = {}
    all_results: list = []

    for i, (ll, ls, lp) in enumerate(model_combos, 1):
        m = PortfolioChallengeModel()
        data = m.load_data(data_dir)
        m._fit(data["X_train"], data["R_train"],
               lambda_linear=ll, lambda_special=ls,
               poly_lambda=lp,
               verbose=False)

        val_pred    = m._fit_history["_val_pred"]
        val_returns = m._fit_history["_val_returns"]

        best_sharpe_this_model = -np.inf
        for mvo_lam, wcap in port_combos:
            sharpe = _portfolio_sharpe_from_preds(
                val_pred, val_returns,
                m.Sigma_inv, m.noise_indices,
                mvo_lam, wcap,
            )
            all_results.append({
                "lambda_linear":  ll,
                "lambda_special": ls,
                "poly_lambda":    lp,
                "mvo_lambda":     mvo_lam,
                "weight_cap":     wcap,
                "val_sharpe":     sharpe,
            })
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {
                    "lambda_linear":  ll,
                    "lambda_special": ls,
                    "poly_lambda":    lp,
                    "mvo_lambda":     mvo_lam,
                    "weight_cap":     wcap,
                }
            if sharpe > best_sharpe_this_model:
                best_sharpe_this_model = sharpe

        print(f"[{i:3d}/{len(model_combos)}] "
              f"λ_lin={ll:<6} λ_spe={ls:<6} λ_poly={lp:<8} "
              f"best val Sharpe={best_sharpe_this_model:+.4f}")

    all_results.sort(key=lambda r: r["val_sharpe"], reverse=True)
    hdr = f"{'λ_lin':<8} {'λ_spe':<8} {'λ_poly':<9} {'mvo_λ':<7} {'cap':<6} {'val Sharpe':>12}"
    sep = "─" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for r in all_results[:15]:
        print(f"{r['lambda_linear']:<8} {r['lambda_special']:<8} "
              f"{r['poly_lambda']:<9} "
              f"{r['mvo_lambda']:<7} {r['weight_cap']:<6} {r['val_sharpe']:>+12.4f}")
    print(sep)
    print(f"\nBest params: {best_params}\n→  val Sharpe = {best_sharpe:+.4f}")

    print("\nRetraining on full training data with best params …")
    best_model = PortfolioChallengeModel()
    best_data  = best_model.load_data(data_dir)
    best_model.fit(
        best_data["X_train"], best_data["R_train"],
        lambda_linear=best_params["lambda_linear"],
        lambda_special=best_params["lambda_special"],
        poly_lambda=best_params["poly_lambda"],
        verbose=True,
    )
    global MVO_LAMBDA, WEIGHT_CAP
    MVO_LAMBDA = best_params["mvo_lambda"]
    WEIGHT_CAP = best_params["weight_cap"]

    submission = best_model.build_submission(best_data["X_train"], best_data["X_test"])
    best_model.save_submission(submission)
    return best_params


def main():
    model = PortfolioChallengeModel()
    data = model.load_data("data")
    model.fit(
        data["X_train"], data["R_train"],
        lambda_linear=0.1,
        lambda_special=1000.0,
        poly_lambda=1000.0,
        verbose=True,
    )
    submission = model.build_submission(data["X_train"], data["X_test"])
    model.save_submission(submission)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "grid":
        grid_search_portfolio(data_dir="data")
    else:
        main()