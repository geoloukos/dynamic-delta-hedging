from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real


def _validate_finite_real(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _validate_positive_real(name: str, value) -> float:
    value = _validate_finite_real(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return value


def _validate_positive_integer(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _validate_spread(name: str, value) -> float:
    value = _validate_finite_real(name, value)
    if not 0.0 <= value < 2.0:
        raise ValueError(f"{name} must satisfy 0 <= {name} < 2.")
    return value


def _validate_seed(seed) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer or None.")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    return seed


def _validate_currency(currency) -> str:
    if not isinstance(currency, str):
        raise TypeError("currency must be a string.")
    currency = currency.strip().upper()
    if not currency:
        raise ValueError("currency cannot be empty.")
    return currency


@dataclass(frozen=True)
class HedgingConfig:
    """Common market, contract and simulation assumptions.

    Pricing functions return model price units. Cash flows in the hedging
    engine are converted to contract-level currency amounts through
    ``contract_multiplier``.
    """

    F0: float
    K: float
    T: float
    r: float
    sigma: float
    futures_spread: float
    n_steps: int
    n_paths: int
    option_spread: float = 0.0
    currency: str = "EUR"
    contract_multiplier: float = 1.0
    mu: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        values = {
            "F0": _validate_positive_real("F0", self.F0),
            "K": _validate_positive_real("K", self.K),
            "T": _validate_positive_real("T", self.T),
            "r": _validate_finite_real("r", self.r),
            "sigma": _validate_positive_real("sigma", self.sigma),
            "futures_spread": _validate_spread("futures_spread", self.futures_spread),
            "option_spread": _validate_spread("option_spread", self.option_spread),
            "n_steps": _validate_positive_integer("n_steps", self.n_steps),
            "n_paths": _validate_positive_integer("n_paths", self.n_paths),
            "contract_multiplier": _validate_positive_real(
                "contract_multiplier", self.contract_multiplier
            ),
            "mu": _validate_finite_real("mu", self.mu),
            "seed": _validate_seed(self.seed),
            "currency": _validate_currency(self.currency),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def dt(self) -> float:
        return self.T / self.n_steps
