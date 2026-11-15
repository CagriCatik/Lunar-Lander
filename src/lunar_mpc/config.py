"""Validated TOML configuration; no simulator dependency."""

import math
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass(frozen=True)
class MPCConfig:
    horizon: int = 30
    dt: float = 0.04
    knots: int = 6
    samples: int = 96
    elites: int = 12
    iterations: int = 3
    replan_interval: int = 3
    noise: float = 0.45
    min_std: float = 0.08

    def __post_init__(self):
        for name in ("horizon", "knots", "samples", "elites", "iterations", "replan_interval"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"mpc.{name} must be a positive integer")
        if self.knots > self.horizon or self.elites > self.samples:
            raise ValueError("Require knots <= horizon and elites <= samples")
        if self.samples < 2:
            raise ValueError("mpc.samples must be at least 2")
        for name in ("dt", "noise", "min_std"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"mpc.{name} must be finite and positive")
        if self.dt < 0.02 or not math.isclose(
            self.dt / 0.02, round(self.dt / 0.02), rel_tol=0, abs_tol=1e-8
        ):
            raise ValueError("mpc.dt must be a multiple of the simulator's 0.02 s timestep")


@dataclass(frozen=True)
class EnvironmentConfig:
    gravity: float = -10.0
    wind: bool = False
    wind_power: float = 15.0
    turbulence_power: float = 1.5
    max_steps: int = 1000

    def __post_init__(self):
        if type(self.gravity) not in (int, float) or not -12 < self.gravity < 0:
            raise ValueError("environment.gravity must be between -12 and 0 (exclusive)")
        if type(self.wind) is not bool:
            raise ValueError("environment.wind must be true or false")
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("environment.max_steps must be a positive integer")
        for name in ("wind_power", "turbulence_power"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"environment.{name} must be finite and nonnegative")


@dataclass(frozen=True)
class Config:
    mpc: MPCConfig = field(default_factory=MPCConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        return Config()
    with Path(path).open("rb") as stream:
        data = tomllib.load(stream)
    unknown = data.keys() - {"mpc", "environment"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    values = {}
    for name, cls in (("mpc", MPCConfig), ("environment", EnvironmentConfig)):
        section = data.get(name, {})
        if not isinstance(section, dict):
            raise ValueError(f"{name} must be a TOML table")
        unknown = section.keys() - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")
        values[name] = cls(**section)
    return Config(**values)
