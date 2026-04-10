from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Plot probabilistic load-displacement envelope.")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "augmentation_physics_fixed" / "load_displacement_full_aug.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "Plottings" / "output" / "load_displacement_envelope.png",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    job_col = "job_aug" if "job_aug" in df.columns else "job"
    u_grid = np.linspace(df["U2"].min(), df["U2"].max(), 300)
    responses = [np.interp(u_grid, g["U2"], g["RF2"]) for _, g in df.groupby(job_col)]
    responses = np.array(responses)

    mean_curve = responses.mean(axis=0)
    p05 = np.percentile(responses, 5, axis=0)
    p95 = np.percentile(responses, 95, axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.fill_between(u_grid, p05, p95, color="steelblue", alpha=0.25, label="5-95%")
    plt.plot(u_grid, mean_curve, color="darkred", linewidth=2.2, label="Mean")
    plt.xlabel("Displacement U2 [mm]")
    plt.ylabel("Reaction Force RF2 [N]")
    plt.title("Probabilistic Load-Displacement Envelope")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)


if __name__ == "__main__":
    main()
