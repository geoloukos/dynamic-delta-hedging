from dataclasses import dataclass

from dynamic_delta_hedging.strategies.asset_tolerance import AssetToleranceStrategy
from dynamic_delta_hedging.strategies.delta_tolerance import DeltaToleranceStrategy
from dynamic_delta_hedging.strategies.fixed_band import FixedBandStrategy
from dynamic_delta_hedging.strategies.fixed_interval import FixedIntervalStrategy
from dynamic_delta_hedging.strategies.ww_inspired import WWInspiredStrategy


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    family: str
    parameter_name: str
    parameter_value: float
    display_name: str


def _spec(strategy_id, family, parameter_name, parameter_value, display_name):
    return StrategySpec(
        strategy_id=strategy_id,
        family=family,
        parameter_name=parameter_name,
        parameter_value=float(parameter_value),
        display_name=display_name,
    )


DEFAULT_STRATEGY_SPECS = (
    _spec("A", "Fixed interval", "interval", 1, "Fixed Interval 1"),
    _spec("B", "Fixed interval", "interval", 2, "Fixed Interval 2"),
    _spec("C", "Fixed interval", "interval", 5, "Fixed Interval 5"),
    _spec("D", "Fixed interval", "interval", 10, "Fixed Interval 10"),
    _spec("E", "Fixed interval", "interval", 21, "Fixed Interval 21"),
    _spec("F", "Fixed band", "band_width", 0.01, "Fixed Band 0.01"),
    _spec("G", "Fixed band", "band_width", 0.02, "Fixed Band 0.02"),
    _spec("H", "Fixed band", "band_width", 0.03, "Fixed Band 0.03"),
    _spec("I", "Fixed band", "band_width", 0.05, "Fixed Band 0.05"),
    _spec("J", "Fixed band", "band_width", 0.10, "Fixed Band 0.10"),
    _spec("K", "WW-inspired", "risk_aversion", 0.25, "WW-inspired λ=0.25"),
    _spec("L", "WW-inspired", "risk_aversion", 0.50, "WW-inspired λ=0.50"),
    _spec("M", "WW-inspired", "risk_aversion", 1.00, "WW-inspired λ=1.00"),
    _spec("N", "WW-inspired", "risk_aversion", 2.00, "WW-inspired λ=2.00"),
    _spec("O", "WW-inspired", "risk_aversion", 4.00, "WW-inspired λ=4.00"),
    _spec("P", "Delta tolerance", "delta_tolerance", 0.01, "Delta Tolerance 0.01"),
    _spec("Q", "Delta tolerance", "delta_tolerance", 0.02, "Delta Tolerance 0.02"),
    _spec("R", "Delta tolerance", "delta_tolerance", 0.03, "Delta Tolerance 0.03"),
    _spec("S", "Delta tolerance", "delta_tolerance", 0.05, "Delta Tolerance 0.05"),
    _spec("T", "Delta tolerance", "delta_tolerance", 0.10, "Delta Tolerance 0.10"),
    _spec("U", "Asset tolerance", "asset_tolerance", 0.015, "Asset Tolerance 1.50%"),
    _spec("V", "Asset tolerance", "asset_tolerance", 0.0225, "Asset Tolerance 2.25%"),
    _spec("W", "Asset tolerance", "asset_tolerance", 0.035, "Asset Tolerance 3.50%"),
    _spec("X", "Asset tolerance", "asset_tolerance", 0.050, "Asset Tolerance 5.00%"),
    _spec("Y", "Asset tolerance", "asset_tolerance", 0.070, "Asset Tolerance 7.00%"),
)

_SPEC_BY_ID = {spec.strategy_id: spec for spec in DEFAULT_STRATEGY_SPECS}


def get_strategy_spec(strategy_id: str) -> StrategySpec:
    if not isinstance(strategy_id, str):
        raise TypeError("strategy_id must be a string.")
    strategy_id = strategy_id.strip().upper()
    try:
        return _SPEC_BY_ID[strategy_id]
    except KeyError as error:
        raise ValueError(f"Unknown strategy_id: {strategy_id!r}.") from error


def build_strategy(spec_or_id):
    spec = get_strategy_spec(spec_or_id) if isinstance(spec_or_id, str) else spec_or_id
    if not isinstance(spec, StrategySpec):
        raise TypeError("spec_or_id must be a strategy ID or StrategySpec.")
    value = spec.parameter_value
    if spec.family == "Fixed interval":
        return FixedIntervalStrategy(interval=int(round(value)))
    if spec.family == "Fixed band":
        return FixedBandStrategy(band_width=value)
    if spec.family == "WW-inspired":
        return WWInspiredStrategy(risk_aversion=value)
    if spec.family == "Delta tolerance":
        return DeltaToleranceStrategy(tolerance=value)
    if spec.family == "Asset tolerance":
        return AssetToleranceStrategy(tolerance=value)
    raise ValueError(f"Unsupported strategy family: {spec.family}.")


