import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

from dynamic_delta_hedging.config import HedgingConfig


def _as_finite_float_array(values: ArrayLike, name: str, ) -> np.ndarray:
    """
    Convert a scalar or array-like input to a
    floating-point NumPy array.

    All values must be finite.
    """

    array = np.asarray(values, dtype=float)

    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must contain only finite values.')

    return array


def _prepare_generalized_bsm_inputs(
    S: ArrayLike,
    K: ArrayLike,
    tau: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    b: ArrayLike,
):
    """
    Convert, broadcast and validate all generalized
    Black-Scholes inputs.

    The inputs may be scalars or broadcast-compatible
    NumPy arrays.
    """

    S_array = _as_finite_float_array(
        S,
        "S",
    )

    K_array = _as_finite_float_array(
        K,
        "K",
    )

    tau_array = _as_finite_float_array(
        tau,
        "tau",
    )

    r_array = _as_finite_float_array(
        r,
        "r",
    )

    sigma_array = _as_finite_float_array(
        sigma,
        "sigma",
    )

    b_array = _as_finite_float_array(
        b,
        "b",
    )

    try:
        (
            S_array,
            K_array,
            tau_array,
            r_array,
            sigma_array,
            b_array,
        ) = np.broadcast_arrays(S_array, K_array, tau_array, r_array, sigma_array, b_array)

    except ValueError as error:
        raise ValueError(
            "S, K, tau, r, sigma and b must be "
            "broadcast-compatible."
        ) from error

    if np.any(S_array <= 0.0):
        raise ValueError(
            "S must be positive."
        )

    if np.any(K_array <= 0.0):
        raise ValueError(
            "K must be positive."
        )

    if np.any(sigma_array <= 0.0):
        raise ValueError(
            "sigma must be positive."
        )

    if np.any(tau_array < 0.0):
        raise ValueError(
            "tau cannot be negative."
        )

    return (S_array, K_array, tau_array, r_array, sigma_array, b_array)


def _return_scalar_or_array(values: np.ndarray, ):
    """
    Return a Python float for scalar calculations and
    a NumPy array for vectorized calculations.
    """

    values = np.asarray(values, dtype=float)

    if values.ndim == 0:
        return float(values)

    return values


def _generalized_bsm_d1(
    S: np.ndarray,
    K: np.ndarray,
    tau: np.ndarray,
    sigma: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate generalized Black-Scholes d1.

    Exact-expiry observations use a temporary safe
    maturity of one. Their d1 values are ignored by the
    public pricing functions and replaced by the correct
    expiry limits.
    """

    positive_tau = (tau > 0.0)

    safe_tau = np.where(positive_tau, tau, 1.0)

    sqrt_tau = np.sqrt(safe_tau)

    d1 = (np.log(S / K) + (b + 0.5 * sigma ** 2) * safe_tau) / (sigma * sqrt_tau)

    return (d1, positive_tau)


# The generalized Black-Scholes functions remain
# independent of the experiment configuration so that
# they can also be reused in future projects.


def generalized_bsm_call_price(
    S: ArrayLike,
    K: ArrayLike,
    tau: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    b: ArrayLike,
):
    """
    Price a European call using the generalized
    Black-Scholes model.

    All inputs may be scalars or broadcast-compatible
    arrays.

    At exact expiry, the option value equals:

        max(S - K, 0)
    """

    (
        S_array,
        K_array,
        tau_array,
        r_array,
        sigma_array,
        b_array,
    ) = _prepare_generalized_bsm_inputs(S=S, K=K, tau=tau, r=r, sigma=sigma, b=b)

    d1, positive_tau = (
        _generalized_bsm_d1(
            S=S_array,
            K=K_array,
            tau=tau_array,
            sigma=sigma_array,
            b=b_array,
        )
    )

    safe_tau = np.where(positive_tau, tau_array, 1.0)

    sqrt_tau = np.sqrt(safe_tau)

    d2 = (d1 - sigma_array * sqrt_tau)

    model_prices = (
        S_array
        * np.exp((b_array - r_array) * safe_tau)
        * norm.cdf(d1)
        - K_array
        * np.exp(-r_array * safe_tau)
        * norm.cdf(d2)
    )

    expiry_payoffs = np.maximum(S_array - K_array, 0.0)

    prices = np.where(positive_tau, model_prices, expiry_payoffs)

    return _return_scalar_or_array(prices)


def generalized_bsm_call_delta(
    S: ArrayLike,
    K: ArrayLike,
    tau: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    b: ArrayLike,
):
    """
    Return the generalized Black-Scholes call delta.

    At exact expiry, the practical convention is:

    - 1.0 when S > K,
    - 0.0 when S < K,
    - 0.5 when S = K.
    """

    (
        S_array,
        K_array,
        tau_array,
        r_array,
        sigma_array,
        b_array,
    ) = _prepare_generalized_bsm_inputs(S=S, K=K, tau=tau, r=r, sigma=sigma, b=b)

    d1, positive_tau = (
        _generalized_bsm_d1(
            S=S_array,
            K=K_array,
            tau=tau_array,
            sigma=sigma_array,
            b=b_array,
        )
    )

    safe_tau = np.where(positive_tau, tau_array, 1.0)

    model_deltas = (np.exp((b_array - r_array) * safe_tau) * norm.cdf(d1))

    expiry_deltas = np.where(S_array > K_array, 1.0, np.where(S_array < K_array, 0.0, 0.5))

    deltas = np.where(positive_tau, model_deltas, expiry_deltas)

    return _return_scalar_or_array(deltas)


def generalized_bsm_call_gamma(
    S: ArrayLike,
    K: ArrayLike,
    tau: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    b: ArrayLike,
):
    """
    Return the generalized Black-Scholes call gamma.

    Gamma measures how quickly call delta changes when
    the underlying price changes.

    The hedging engine does not open a new hedge after
    maturity, so gamma is defined as zero at exact
    expiry as a practical numerical convention.
    """

    (
        S_array,
        K_array,
        tau_array,
        r_array,
        sigma_array,
        b_array,
    ) = _prepare_generalized_bsm_inputs(S=S, K=K, tau=tau, r=r, sigma=sigma, b=b)

    d1, positive_tau = (
        _generalized_bsm_d1(
            S=S_array,
            K=K_array,
            tau=tau_array,
            sigma=sigma_array,
            b=b_array,
        )
    )

    safe_tau = np.where(positive_tau, tau_array, 1.0)

    model_gammas = (
        np.exp((b_array - r_array) * safe_tau)
        * norm.pdf(d1)
        / (S_array * sigma_array * np.sqrt(safe_tau))
    )

    gammas = np.where(positive_tau, model_gammas, 0.0)

    return _return_scalar_or_array(gammas)


# Setting b = 0 gives the Black-76 model for European
# options on futures.
#
# F and tau remain explicit because they change during
# the life of the option.
#
# K, r and sigma are read from the common configuration.


def black76_call_price(F: ArrayLike, tau: ArrayLike, config: HedgingConfig, ):
    """
    Price a European call on futures using Black-76.

    F and tau may be scalars or broadcast-compatible
    arrays.
    """

    return generalized_bsm_call_price(
        S=F,
        K=config.K,
        tau=tau,
        r=config.r,
        sigma=config.sigma,
        b=0.0,
    )


def black76_call_delta(F: ArrayLike, tau: ArrayLike, config: HedgingConfig, ):
    """
    Return the Black-76 call delta with respect to the
    futures price.

    F and tau may be scalars or broadcast-compatible
    arrays.
    """

    return generalized_bsm_call_delta(
        S=F,
        K=config.K,
        tau=tau,
        r=config.r,
        sigma=config.sigma,
        b=0.0,
    )


def black76_call_gamma(F: ArrayLike, tau: ArrayLike, config: HedgingConfig, ):
    """
    Return the Black-76 call gamma with respect to the
    futures price.

    F and tau may be scalars or broadcast-compatible
    arrays.
    """

    return generalized_bsm_call_gamma(
        S=F,
        K=config.K,
        tau=tau,
        r=config.r,
        sigma=config.sigma,
        b=0.0,
    )