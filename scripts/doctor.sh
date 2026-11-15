#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
exec "$PYTHON" - <<'PY'
import importlib.metadata
import platform
import sys
import gymnasium as gym
print(f"Python {platform.python_version()} | {platform.system()} {platform.machine()}")
print(f"Interpreter: {sys.executable}")
for package in ("lunar-lander-mpc", "gymnasium", "numpy", "Box2D", "pygame"):
    print(f"{package}: {importlib.metadata.version(package)}")
env = gym.make("LunarLander-v3", continuous=True)
try:
    obs, _ = env.reset(seed=0)
    env.step(env.action_space.sample())
    print(f"Box2D reset/step OK | observation {obs.shape} | actions {env.action_space}")
finally:
    env.close()
PY
