from pathlib import Path
import argparse

from physicsgate.preprocessing import preprocess_pv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/pv.csv"))
    args = parser.parse_args()
    print(*preprocess_pv(args.input, args.output), sep="\n")


if __name__ == "__main__":
    main()
