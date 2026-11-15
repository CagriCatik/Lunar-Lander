# Controller and model

## Boundaries

The controller accepts an eight-value observation and returns a two-value action.
It cannot call `env.step`, copy a Box2D world, access the simulator RNG or inspect
future wind. Its predictions come exclusively from the nominal model. The runner
owns the simulator and writes observations, actions, predictions and metrics.

`model.py`, `guidance.py`, `observer.py`, `controller.py` and `robust.py` depend on
NumPy and validated settings.
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

Original guidance targets an angle from horizontal position and velocity, bounded to ±0.45
radians. Adaptive guidance adds `atan2(estimated_ax, max(3, -gravity-estimated_ay))`
and permits ±0.55 radians. Its angular feedback subtracts estimated angular acceleration;
vertical thrust compensates the estimated vertical acceleration bias. Desired
descent speed depends on height above a lateral-offset-dependent
clearance target and slows near touchdown. PD feedback turns these targets into
main and lateral engine commands.

CEM optimises a sequence of two-component **additive command biases**, divided into
five knots by default. Candidate trajectories recompute feedback from each predicted
state. This is closed-loop/feedback shooting, rather than sampling independent raw
actions for every future physics tick.

Each optimisation starts with zero mean and standard deviation `noise`. Sampled
biases are clipped to [-1, 1]; the sum of feedback and bias is clipped again before
engine mapping. The unmodified feedback trajectory and best candidate found so far
remain in each population. After ranking, the elite sample mean and standard
deviation form the next distribution; `min_std` prevents immediate collapse.

The default horizon has 20 intervals of 0.06 s. Adaptive prediction uses one force
evaluation per interval, freezing accelerations and correcting position integration
to match repeated 0.02 s semi-implicit updates under constant acceleration:

\[
p_{k+1}=p_k+\Delta t\,v_k+\tfrac12\Delta t(\Delta t+0.02)a_k
\]

Angle uses the equivalent correction. Force direction changes inside an interval
are approximated; a test checks exact agreement for constant acceleration. This
reduces repeated NumPy force evaluations. Actual Box2D physics still runs at 50 Hz.
The original controller retains fine substeps, including in the v0.1 comparison.

The first selected bias is held for three simulator steps; feedback and the observer
still update every step. Optimisation restarts from the current measurement, without
a shifted-sequence warm start.

When nominal hover demand lies below the engine's 50% minimum, adaptive guidance
switches the engine around zero desired vertical acceleration instead of using the
normal 25%-power rounding threshold. This on/off velocity feedback avoids the upward
descent bias that otherwise causes low-gravity hovering. Actual commands continue to
obey the simulator's engine dead zones.

## Causal disturbance observer

The observer compares the current measured `[vx, vy, omega]` with the previous
nominal one-step prediction, using the **previous applied action**:

\[
r_k=(v_k-[f(s_{k-1},u_{k-1})]_v)/0.02
\]

The first eight transitions are ignored to reduce initial leg-articulation errors.
Subsequent residuals are clipped to three times the configured acceleration bounds,
then exponentially filtered. Horizontal and angular gains are
`1-exp(-0.02/observer_tau)`; the vertical channel uses twice the time constant because
main-engine exhaust dispersion is particularly noisy. The filtered estimate is
clipped again to the configured bounds. It estimates **combined model error and
disturbance**, not a direct measurement of pure wind.

Predictions add estimated `[ax, ay, angular_acceleration]` to nominal accelerations.
At leg contact, estimation history and biases are cleared, so impact impulses cannot
be learned as wind. A bounce triggers fresh planning instead of reusing a flight bias.
No body properties, future simulator state or wind RNG are read.

## Scenario objective

Every candidate bias sequence is evaluated under five constant acceleration
hypotheses: the central estimate, horizontal acceleration ±`wind_margin`, and angular
acceleration ±`torque_margin`. Feedback compensates the central estimate in every
scenario; state differences drive each simulated feedback response.

\[
J=(1-\rho)\operatorname{mean}_s J_s+\rho\max_s J_s
\]

Here `rho=robust.risk_weight`. The feedback-only and best-so-far candidates remain
available under this same scenario objective. Only the central trajectory is drawn.
These five samples do not cover all combinations, time-varying gusts, exhaust noise
or contact physics. There is no calibrated confidence interval or robust-invariance
proof associated with the margins.

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

On either leg contact, adaptive MPC clears its prediction, disturbance history and
held bias. The main engine is off. A separate attitude stabiliser commands the side
jet from `(16*angle+6*omega)/4.9`, bounded to [-1, 1] with the engine's minimum firing
level. It releases the jets when `abs(angle)<0.06` and `abs(omega)<0.25`, allowing
Box2D to sleep. This can arrest first-leg tipping that the old engine-off policy
could not. The original nominal/feedback controllers retain `[-1, 0]` contact mode.
Ground contact remains excluded from flight prediction; the simulator determines
whether the vehicle settles, bounces or crashes.

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
targeted landings used by CI. Additional tests cover known disturbance recovery,
noise rejection, outlier saturation, scenario aggregation, compensation signs, contact
reset/bounce recovery, low-gravity engine switching, nominal-controller compatibility
and two wind seeds that exercise the new landing behaviour.

Original-controller compatibility is checked on fixed observations against action
fixtures captured from the v0.1 commit. End-to-end Box2D checks assert a successful
landing and score threshold rather than a Linux-specific exact episode reward;
macOS/ARM can produce a different trajectory with the same seed.
