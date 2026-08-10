import numpy as np

from dynamic_delta_hedging.config import HedgingConfig


def _validate_standard_normal_shocks(standard_normal_shocks, config: HedgingConfig) -> np.ndarray:
    shocks = np.asarray(standard_normal_shocks, dtype=float)
    if shocks.ndim != 2:
        raise ValueError("standard_normal_shocks must be two-dimensional.")
    if shocks.shape[1] != config.n_steps:
        raise ValueError(
            f"standard_normal_shocks must contain {config.n_steps} shocks per path."
        )
    if shocks.shape[0] == 0:
        raise ValueError("standard_normal_shocks must contain at least one path.")
    if not np.all(np.isfinite(shocks)):
        raise ValueError("standard_normal_shocks must contain only finite values.")
    return shocks


def _validate_realized_sigma(realized_sigma, n_paths: int, config: HedgingConfig) -> np.ndarray:
    if realized_sigma is None:
        return np.full(n_paths, config.sigma, dtype=float)

    sigma = np.asarray(realized_sigma, dtype=float)
    try:
        sigma = np.broadcast_to(sigma, (n_paths,))
    except ValueError as error:
        raise ValueError(
            "realized_sigma must be scalar or broadcast-compatible with the path batch."
        ) from error
    if not np.all(np.isfinite(sigma)):
        raise ValueError("realized_sigma must contain only finite values.")
    if np.any(sigma <= 0.0):
        raise ValueError("realized_sigma must be positive.")
    return np.array(sigma, dtype=float, copy=True)


def simulate_futures_paths_from_shocks(
    config: HedgingConfig,
    standard_normal_shocks,
    *,
    realized_sigma=None,
) -> np.ndarray:
    """Generate GBM futures paths from supplied shocks.

    ``config.sigma`` remains the pricing/hedging-model volatility.  The optional
    ``realized_sigma`` controls only the volatility used to generate the actual
    futures paths.  It may be one scalar or one value per path.
    """
    if not isinstance(config, HedgingConfig):
        raise TypeError("config must be a HedgingConfig.")

    shocks = _validate_standard_normal_shocks(standard_normal_shocks, config)
    n_paths = shocks.shape[0]
    sigmas = _validate_realized_sigma(realized_sigma, n_paths, config)

    dt = config.dt
    drift = (config.mu - 0.5 * sigmas ** 2) * dt
    diffusion = sigmas * np.sqrt(dt)

    # Vectorize across paths while stepping through time.  This avoids allocating
    # two additional full path-by-step matrices for log increments and cumulative
    # log levels, which matters for the large campaign sample.
    futures_paths = np.empty((n_paths, config.n_steps + 1), dtype=float)
    futures_paths[:, 0] = config.F0
    for step in range(config.n_steps):
        futures_paths[:, step + 1] = futures_paths[:, step] * np.exp(
            drift + diffusion * shocks[:, step]
        )
    return futures_paths


def simulate_futures_paths(config: HedgingConfig) -> np.ndarray:
    """Simulate futures price paths using geometric Brownian motion."""
    rng = np.random.default_rng(config.seed)
    random_shocks = rng.standard_normal((config.n_paths, config.n_steps))

    # Preserve the historical scalar-sigma research path construction exactly.
    futures_paths = np.empty((config.n_paths, config.n_steps + 1))
    futures_paths[:, 0] = config.F0
    for i in range(config.n_steps):
        futures_paths[:, i + 1] = (
            futures_paths[:, i]
            * np.exp(
                (config.mu - 0.5 * config.sigma ** 2) * config.dt
                + config.sigma * np.sqrt(config.dt) * random_shocks[:, i]
            )
        )
    return futures_paths
