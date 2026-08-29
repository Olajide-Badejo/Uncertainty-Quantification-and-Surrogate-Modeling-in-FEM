from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Plot individual load-displacement curves.")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "augmentation_physics_fixed" / "load_displacement_full_aug.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "Plottings" / "output" / "load_displacement_individual",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    job_col = "job_aug" if "job_aug" in df.columns else "job"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for job, g in df.groupby(job_col):
        plt.figure(figsize=(7, 4.5))
        plt.plot(g["U2"], g["RF2"], color="darkred", linewidth=2.0)
        plt.xlabel("Displacement U2 [mm]")
        plt.ylabel("Reaction Force RF2 [N]")
        plt.title(f"Load-Displacement: {job}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(args.output_dir / f"{job}_ld.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
