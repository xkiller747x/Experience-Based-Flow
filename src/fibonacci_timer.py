"""Print Fibonacci terms over a fixed total duration."""

from __future__ import annotations

import argparse
import time


def fibonacci_sequence(count: int) -> list[int]:
    """Return the first ``count`` Fibonacci numbers."""
    if count <= 0:
        return []
    if count == 1:
        return [0]

    sequence = [0, 1]
    while len(sequence) < count:
        sequence.append(sequence[-1] + sequence[-2])
    return sequence


def print_fibonacci_over_duration(count: int, duration_seconds: float) -> None:
    """Print Fibonacci numbers, finishing in roughly ``duration_seconds``."""
    if count <= 0:
        raise ValueError("count must be greater than 0")
    if duration_seconds < 0:
        raise ValueError("duration_seconds must be non-negative")

    sequence = fibonacci_sequence(count)
    start = time.monotonic()

    for index, value in enumerate(sequence):
        elapsed = time.monotonic() - start
        print(f"Term {index + 1:02d}: {value}  (elapsed {elapsed:.2f}s)")

        if index == count - 1:
            break

        if count == 1:
            continue

        target_elapsed = duration_seconds * (index + 1) / (count - 1)
        sleep_for = target_elapsed - (time.monotonic() - start)
        if sleep_for > 0:
            time.sleep(sleep_for)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print Fibonacci numbers within a fixed total duration."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of terms to print. Default: 50.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Total runtime in seconds. Default: 60.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_fibonacci_over_duration(args.count, args.duration)


if __name__ == "__main__":
    main()
