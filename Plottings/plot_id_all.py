from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Plot all load-displacement curves.")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "augmentation_physics_fixed" / "load_displacement_full_aug.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "Plottings" / "output" / "load_displacement_all.png",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    job_col = "job_aug" if "job_aug" in df.columns else "job"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    for _, g in df.groupby(job_col):
        plt.plot(g["U2"], g["RF2"], alpha=0.25, linewidth=1.0)
    plt.xlabel("Displacement U2 [mm]")
    plt.ylabel("Reaction Force RF2 [N]")
    plt.title("Load-Displacement Curves (All Samples)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)


if __name__ == "__main__":
    main()
