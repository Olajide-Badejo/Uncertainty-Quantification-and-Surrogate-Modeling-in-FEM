from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Plot individual compression damage curves.")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "augmentation_physics_fixed" / "crack_evolution_full_aug.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "Plottings" / "output" / "compression_damage_individual",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    job_col = "job_aug" if "job_aug" in df.columns else "job"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for job, g in df.groupby(job_col):
        plt.figure(figsize=(7, 4.5))
        plt.plot(g["U2"], g["DAMAGEC_max"], color="steelblue", linewidth=2.2)
        plt.xlabel("Displacement U2 [mm]")
        plt.ylabel("Compression Damage [-]")
        plt.title(f"Compression Damage: {job}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(args.output_dir / f"{job}_compression_damage.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
