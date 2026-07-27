from pathlib import Path
import argparse

from physicsgate.preprocessing import preprocess_estm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/estm.csv"))
    args = parser.parse_args()
    print(*preprocess_estm(args.input, args.output), sep="\n")


if __name__ == "__main__":
    main()
