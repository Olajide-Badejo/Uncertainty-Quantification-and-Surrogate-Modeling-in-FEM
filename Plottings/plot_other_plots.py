from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Plot peak-load and failure-displacement histograms.")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "augmentation_physics_fixed" / "load_displacement_full_aug.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "Plottings" / "output" / "distributions",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    job_col = "job_aug" if "job_aug" in df.columns else "job"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    peak = df.groupby(job_col)["RF2"].max()
    plt.figure()
    peak.hist(bins=15, edgecolor="black")
    plt.xlabel("Peak Load [N]")
    plt.ylabel("Frequency")
    plt.title("Distribution of Peak Load Capacity")
    plt.tight_layout()
    plt.savefig(args.output_dir / "peak_load_distribution.png", dpi=300)

    fail_disp = df.groupby(job_col)["U2"].max()
    plt.figure()
    fail_disp.hist(bins=15, color="orange", edgecolor="black")
    plt.xlabel("Failure Displacement [mm]")
    plt.ylabel("Frequency")
    plt.title("Distribution of Failure Displacement")
    plt.tight_layout()
    plt.savefig(args.output_dir / "failure_displacement_distribution.png", dpi=300)


if __name__ == "__main__":
    main()
