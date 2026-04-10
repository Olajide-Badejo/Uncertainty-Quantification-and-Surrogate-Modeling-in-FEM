from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _envelope(df: pd.DataFrame, job_col: str, u_grid: np.ndarray, var: str):
    curves = [np.interp(u_grid, g["U2"], g[var]) for _, g in df.groupby(job_col)]
    curves = np.array(curves)
    return curves.mean(axis=0), np.percentile(curves, 95, axis=0)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Plot damage envelopes.")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "augmentation_physics_fixed" / "crack_evolution_full_aug.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "Plottings" / "output" / "damage_envelope.png",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    job_col = "job_aug" if "job_aug" in df.columns else "job"
    u_grid = np.linspace(df["U2"].min(), df["U2"].max(), 200)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    for var, label in [("DAMAGET_max", "Tension Damage"), ("SDEG_max", "Stiffness Degradation")]:
        mean, p95 = _envelope(df, job_col, u_grid, var)
        plt.plot(u_grid, mean, label=f"{label} (mean)")
        plt.plot(u_grid, p95, "--", label=f"{label} (95%)")
    plt.xlabel("Displacement U2 [mm]")
    plt.ylabel("Damage Index [-]")
    plt.title("Probabilistic Damage Evolution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)


if __name__ == "__main__":
    main()
