#!/usr/bin/env python3
"""Periodically display Orange Pi temperatures and warn on sustained heat."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path


THERMAL_ROOT = Path("/sys/class/thermal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor Orange Pi thermal zones and warn on sustained heat."
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=80.0,
        help="warning temperature in Celsius (default: 80)",
    )
    parser.add_argument(
        "-i", "--interval", type=float, default=5.0,
        help="sampling interval in seconds (default: 5)",
    )
    parser.add_argument(
        "-c", "--consecutive", type=int, default=3,
        help="consecutive hot samples required for a warning (default: 3)",
    )
    parser.add_argument(
        "--once", action="store_true", help="print one sample and exit",
    )
    parser.add_argument(
        "--thermal-root", type=Path, default=THERMAL_ROOT,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than 0")
    if args.consecutive <= 0:
        parser.error("--consecutive must be greater than 0")
    return args


def read_temperatures(root: Path) -> list[tuple[str, float]]:
    readings: list[tuple[str, float]] = []
    for zone in sorted(root.glob("thermal_zone*")):
        try:
            raw = float((zone / "temp").read_text().strip())
            zone_type = (zone / "type").read_text().strip()
        except (OSError, ValueError):
            continue

        # The kernel normally exposes millidegrees Celsius, although a few
        # drivers expose degrees directly.
        temperature = raw / 1000.0 if abs(raw) >= 1000 else raw
        readings.append((zone_type or zone.name, temperature))
    return readings


def format_readings(readings: list[tuple[str, float]]) -> str:
    return "  ".join(f"{name}: {temperature:.1f}°C" for name, temperature in readings)


def main() -> int:
    args = parse_args()
    running = True

    def stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    hot_count = 0
    warning_active = False
    print(
        f"Temperature monitor started: threshold={args.threshold:.1f}°C, "
        f"interval={args.interval:g}s, consecutive samples={args.consecutive}"
    )

    while running:
        readings = read_temperatures(args.thermal_root)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not readings:
            print(
                f"[{timestamp}] ERROR: no readable thermal zones under "
                f"{args.thermal_root}",
                file=sys.stderr,
                flush=True,
            )
            return 1

        hottest_name, hottest = max(readings, key=lambda item: item[1])
        print(f"[{timestamp}] {format_readings(readings)}", flush=True)

        if hottest >= args.threshold:
            hot_count += 1
            if hot_count >= args.consecutive:
                duration = hot_count * args.interval
                prefix = "\033[1;31m" if sys.stderr.isatty() else ""
                suffix = "\033[0m" if prefix else ""
                print(
                    f"{prefix}[WARNING] {hottest_name} is {hottest:.1f}°C; "
                    f"temperature has stayed at or above {args.threshold:.1f}°C "
                    f"for {duration:g}s.{suffix}",
                    file=sys.stderr,
                    flush=True,
                )
                warning_active = True
        else:
            if warning_active:
                print(
                    f"[RECOVERED] Temperature is below {args.threshold:.1f}°C.",
                    flush=True,
                )
            hot_count = 0
            warning_active = False

        if args.once:
            break
        time.sleep(args.interval)

    print("Temperature monitor stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
