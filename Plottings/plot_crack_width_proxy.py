from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Plot crack-width proxy curves.")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "augmentation_physics_fixed" / "crack_evolution_full_aug.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "Plottings" / "output" / "crack_width_proxy.png",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    job_col = "job_aug" if "job_aug" in df.columns else "job"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    for _, g in df.groupby(job_col):
        plt.plot(g["U2"], g["PEEQ_max"], alpha=0.3)
    plt.xlabel("Displacement U2 [mm]")
    plt.ylabel("Max PEEQ (proxy)")
    plt.title("Crack Width Proxy vs Displacement")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)


if __name__ == "__main__":
    main()
