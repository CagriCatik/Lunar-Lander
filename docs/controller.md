# Controller and model

## Boundaries

The controller accepts an eight-value observation and returns a two-value action.
It cannot call `env.step`, copy a Box2D world, access the simulator RNG or inspect
future wind. Its predictions come exclusively from the nominal model. The runner
owns the simulator and writes observations, actions, predictions and metrics.

`model.py`, `guidance.py` and `controller.py` depend on NumPy and validated settings.
Rendering and file output remain outside those modules. Model prediction, guidance,
candidate rollout and population refinement each have their own function boundary.

## State conversion

For Gymnasium 1.2.3's default viewport and 50 Hz physics:

| Observation element | Physical state | Multiply by |
|---|---|---:|
| `obs[0]` | Horizontal position relative to pad centre | 10 |
| `obs[1]` | Body-origin height relative to nominal landing target | 20/3 |
| `obs[2]` | Horizontal velocity | 5 |
| `obs[3]` | Vertical velocity | 7.5 |
| `obs[4]` | Body angle, radians | 1 |
| `obs[5]` | Angular velocity, radians/s | 2.5 |
| `obs[6:8]` | Leg-contact flags | Boolean mode inputs |

These are **Box2D units**, not calibrated real spacecraft measurements. The target
height is the helipad height plus `LEG_DOWN / SCALE`; it is not the lander's centre
of mass. The nominal model approximates body-origin and centre-of-mass velocities.

## Engine mapping

For clipped actions `a_main` and `a_side`:

\[
p_m = \begin{cases}(a_\mathrm{main}+1)/2 & a_\mathrm{main}>0\\0 & \text{otherwise}\end{cases}
\]

\[
p_s = \begin{cases}a_\mathrm{side} & |a_\mathrm{side}|>0.5\\0 & \text{otherwise}\end{cases}
\]

Thus `main=0` is **off**, and `side=±0.5` is also **off**. The main-engine endpoint
is taken from the pinned source, which is more precise than the prose documentation.
The signed side power is its action magnitude and direction, not an affine remapping
from zero power. Tests exercise the exact endpoints.

## Nominal dynamics

With `s = [x, y, vx, vy, theta, omega]`, a default mass of 4.958889 (body plus legs),
an approximate composite inertia of 0.90, and a 0.02 s integration step:

\[
F_m = 50(4/30)(13)p_m, \qquad F_s = 50(12/30)(0.6)p_s
\]

\[
\dot v_x = (-F_m\sin\theta+F_s\cos\theta)/m
\]

\[
\dot v_y = (F_m\cos\theta+F_s\sin\theta)/m+g
\]

\[
\dot\omega = -F_s\left[(14/30)\cos^2\theta+(17/30)\sin^2\theta-0.101307\right]/I
\]

Positions and angle use semi-implicit Euler updates. Multiplying the simulator's
nominal engine impulses by 50 gives the equivalent average force. The side-engine
lever arm retains Gymnasium's asymmetric 17/30 and 14/30 placement, with an approximate
body-local centre-of-mass correction. Leg movement, its changing composite inertia,
random exhaust impulse and exhaust-induced torque remain model errors.

Tests compare a one-step prediction with actual free flight after the legs settle,
using bounded error tolerances rather than asserting an exact simulator match.

## Guidance and CEM

Guidance targets an angle from horizontal position and velocity, bounded to ±0.45
radians. Desired descent speed depends on height above a lateral-offset-dependent
clearance target and slows near touchdown. PD feedback turns these targets into
main and lateral engine commands.

CEM optimises a sequence of two-component **additive command biases**, divided into
six knots by default. Candidate trajectories recompute feedback from each predicted
state. This is closed-loop/feedback shooting, rather than sampling independent raw
actions for every future physics tick.

Each optimisation starts with zero mean and standard deviation `noise`. Sampled
biases are clipped to [-1, 1]; the sum of feedback and bias is clipped again before
engine mapping. The unmodified feedback trajectory and best candidate found so far
remain in each population. After ranking, the elite sample mean and standard
deviation form the next distribution; `min_std` prevents immediate collapse.

The default horizon has 30 intervals of 0.04 s. Each interval integrates two physical
steps. The selected first bias is held for three simulator steps; feedback still
updates every step. Optimisation then starts again from the measured state. There
is currently no shifted-sequence warm start, adaptive model or disturbance estimator.

## Objective

For predicted states at each interval, the stage cost is:

| Term | Weight |
|---|---:|
| Horizontal position squared | 0.15 |
| Horizontal velocity squared | 0.25 |
| Vertical velocity tracking error squared | 2.0 |
| Angle tracking error squared | 10.0 |
| Angular velocity squared | 0.3 |
| Effective main power | 0.04 |
| Absolute effective side power | 0.01 |
| Squared penetration below target height | 200 |
| Squared horizontal excursion beyond ±9 units | 100 |
| Squared tilt beyond ±0.7 radians | 100 |

The sum is multiplied by the prediction interval duration. Terminal cost adds
`0.5*x² + 0.2*y²`. These weights live in `_rollout`; they are not Gymnasium rewards.
There is no finite-fuel constraint, terrain map, hard state constraint or collision
guarantee. CEM returns the lowest-cost sampled sequence, not a certified global optimum.

## Contact and evaluation

On either leg contact, the controller clears its displayed prediction and returns
`[-1, 0]`. Ground contact is intentionally excluded from free-flight prediction.
The simulator determines whether the vehicle settles, bounces or crashes.

The outcome classifier checks a body crash first, then horizontal bounds. A terminal,
sleeping vehicle on the pad with at least one reported leg contact is `landed`;
other resting outcomes are `rested_elsewhere`. Requiring both contact flags would
misclassify some stable resting states because the simulator's contact flags are
not a geometrical support estimate. Time-limit truncation remains `timeout`.

The fuel proxy is `sum((main_power + 0.1*abs(side_power))*0.02)`.
Latency includes controller calls only; it excludes physics, rendering and file I/O.
Planning and all-step means are reported separately, along with p95/max control
latency and the fraction exceeding one 20 ms physics step.

## Verification

The tests stress malformed configuration, engine dead-zone boundaries, invalid
observations, input mutation, batched model predictions, seeded optimiser behaviour,
contact-mode switching, optimiser baseline retention, reset state, free-flight model
error, artifact creation, timeout/crash precedence and a complete seeded landing.
The reference multi-seed benchmark checks actual closed-loop behaviour beyond the
single landing used by CI.
