# Adding a model

METIS models implement `RepresentationModel` (`src/metis/models/base.py`),
which *is* the preprocessing `Transform` contract plus a `latent_dim`:

| method | meaning |
|---|---|
| `fit(X) -> self` | learn from **training rows only** |
| `transform(X) -> Z` | map to the latent `(n, latent_dim)` |
| `inverse_transform(Z) -> X_hat` | reconstruct |
| `is_fitted` (property) | bool |
| `get_params()` / `from_params(cls, d)` | JSON-safe round trip |
| `reconstruction_mse(X)` | provided by the base |

`get_params` / `from_params` make `save` / `load` / `to_dict` / `from_dict`
(from `metis.data.preprocessing`) work for free — subclasses self-register
by class name.

## A numpy model

Follow `metis/models/pca.py`: store the fitted state as arrays, implement
the six methods, serialise arrays as lists in `get_params`. Add it to
`metis/models/__init__.py`.

## A torch model

Follow `metis/models/autoencoder.py`:

- Keep torch imports **inside** methods so importing `metis.models` stays
  torch-free (CI installs only `[dev]`).
- Build the `nn.Module` lazily in `fit` (you need `n_features`), store it
  as `self.module` with a `forward` that reconstructs its input.
- Standardise inside `fit` with a `StandardScaler` fitted on the passed
  rows (train-only) and invert it in `inverse_transform`.
- Don't run the optimisation loop yourself — hand `self` and the scaled
  data to `metis.training.Trainer`:

```python
def fit(self, X, *, run=None, trainer=None):
    ...
    self.module = _make_net(...)
    self.history_ = (trainer or Trainer()).fit(self, Xs, run=run)
    return self
```

`Trainer` gives you: fixed-seed reproducibility, full-batch Adam,
early-stopping with best-state restore, an optional best checkpoint, and
optional per-step MLflow logging via a `metis.tracking` run handle.

- `get_params` serialises config + the fitted `StandardScaler` params +
  `module.state_dict()` (as lists); `from_params` rebuilds the net and
  `load_state_dict`s.

## Evaluate it honestly

Compare against a baseline (usually `PCARepresentation`) through the same
instruments:

```python
from metis.evaluation.representation import evaluate_representation, latent_stability
from metis.evaluation.assessment import assess_model

cand = evaluate_representation(model.transform(Xtr), Pb, Thw, Z_ood=..., ...)
base = evaluate_representation(pca.transform(Xtr), Pb, Thw, Z_ood=..., ...)
verdict = assess_model(cand, base, ml_metrics=[...], physical_metrics=[...])
```

`assess_model` only calls a model **better** if an ML metric improves
*and* no physical diagnostic regresses. If a nonlinear model just
reproduces the PCA subspace (check `latent_stability` ≈ 0 across seeds
and the `latent_physical_correlation`s), **record that and stop** — see
`FINDINGS.md` §6 for the worked example.

## Register a validated model

```bash
metis registry add my_model --kind model --run-id <mlflow-id> \
    --dataset-id regime_compact_v1 --scope "what it's approved to reconstruct"
metis registry promote my_model        # needs a scope
```
