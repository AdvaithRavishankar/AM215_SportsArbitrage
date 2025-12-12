"""Profile the main pipeline to identify performance bottlenecks."""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
from pathlib import Path

# Add src to the path so we can import run.py when executing from repo root
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import run  # type: ignore  # noqa: E402


def profile_run(
    output: str = "profile_stats.prof", sort: str = "cumulative", limit: int = 50
) -> str:
    """Run cProfile on run.main and return a printable stats summary.

    Args:
        output: Path to write the raw cProfile stats file (for tools like snakeviz).
        sort: Sort column for pstats (e.g., "cumulative", "tottime").
        limit: Number of rows to show in the printed summary.

    Returns:
        String containing the formatted profiling summary.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    run.main()
    profiler.disable()

    profiler.dump_stats(output)

    buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=buffer).sort_stats(sort)
    stats.print_stats(limit)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the sports arbitrage pipeline")
    parser.add_argument(
        "--output",
        default="profile_stats.prof",
        help="Path to write raw cProfile stats for offline inspection",
    )
    parser.add_argument(
        "--sort",
        default="cumulative",
        choices=["cumulative", "tottime", "time", "calls", "pcalls"],
        help="Sort column for printed stats",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of rows to display in the printed stats",
    )

    args = parser.parse_args()
    summary = profile_run(output=args.output, sort=args.sort, limit=args.limit)
    print(summary)
    print(f"Raw cProfile data written to: {args.output}")


if __name__ == "__main__":
    main()
