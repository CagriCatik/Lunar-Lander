"""Headless episode diagnostics; pyplot is imported only when requested."""

import numpy as np


def plot_episode(directory, summary, rows, predictions):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.asarray(rows)
    t = data[:, 1]
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        ax = axes[0, 0]
        ax.plot(data[:, 2], data[:, 3], color="#63d8ff", label="Actual flight")
        for i, prediction in enumerate(predictions[:: max(1, len(predictions) // 12)]):
            p = np.asarray(prediction["states"])
            ax.plot(
                p[:, 0],
                p[:, 1],
                color="#ffe277",
                alpha=0.45,
                label="MPC predictions" if i == 0 else None,
            )
        ax.scatter([0], [0], marker="x", s=100, color="#7ef2a0", label="Pad target")
        ax.set(xlabel="Horizontal position [Box2D units]", ylabel="Height above target")
        ax.legend()
        ax = axes[0, 1]
        ax.plot(t, data[:, 4], label="vx")
        ax.plot(t, data[:, 5], label="vy")
        ax.set(xlabel="Time [s]", ylabel="Velocity [units/s]")
        ax.legend()
        ax = axes[1, 0]
        ax.plot(t, np.rad2deg(data[:, 6]), label="Angle [deg]")
        ax.plot(t, np.rad2deg(data[:, 7]), label="Angular velocity [deg/s]")
        ax.set(xlabel="Time [s]")
        ax.legend()
        ax = axes[1, 1]
        ax.step(t, data[:, 12], label="Main power", where="post")
        ax.step(t, data[:, 13], label="Signed lateral power", where="post", alpha=0.7)
        ax.set(xlabel="Time [s]", ylabel="Engine power", ylim=(-1.1, 1.1))
        ax.legend()
        for ax in axes.flat:
            ax.grid(alpha=0.15)
        fig.suptitle(
            f"Lunar Lander · {summary['controller'].upper()} · seed {summary['seed']} · "
            f"{summary['outcome']} · reward {summary['reward']:.1f}"
        )
        fig.savefig(directory / "flight.png", dpi=150)
        plt.close(fig)
