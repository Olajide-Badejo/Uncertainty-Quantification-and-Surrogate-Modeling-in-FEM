from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Plot probabilistic crack-width envelope.")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "augmentation_physics_fixed" / "crack_evolution_full_aug.csv",
    )
    parser.add_argument(
        "--element-size",
        type=float,
        default=0.025,
        help="Element size in meters for crack-width proxy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "Plottings" / "output" / "probabilistic_crack_width_envelope.png",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input).copy()
    job_col = "job_aug" if "job_aug" in df.columns else "job"
    df["crack_width"] = df["PEEQ_max"] * args.element_size

    disp_grid = np.linspace(0.0, float(df["U2"].max()), 100)
    curves = [np.interp(disp_grid, g["U2"], g["crack_width"]) for _, g in df.groupby(job_col)]
    curves = np.array(curves)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.fill_between(
        disp_grid,
        np.percentile(curves, 5, axis=0),
        np.percentile(curves, 95, axis=0),
        alpha=0.3,
        label="5-95%",
    )
    plt.plot(disp_grid, curves.mean(axis=0), label="Mean")
    plt.xlabel("Displacement U2 [mm]")
    plt.ylabel("Crack Width Proxy [m]")
    plt.title("Probabilistic Crack Width Evolution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)


if __name__ == "__main__":
    main()
