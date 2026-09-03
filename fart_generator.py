"""Standalone fart-related string generator. No project dependencies."""

import argparse
import random
from typing import Optional


class FartGenerator:
    """Generates short, silly fart-related strings in a chosen style."""

    _STYLES = {
        "default": [
            "PFFFT.",
            "A fart happened.",
            "*toot*",
            "There it is.",
            "Unscheduled gas release detected.",
        ],
        "polite": [
            "Excuse me.",
            "Pardon the disturbance.",
            "Ahem... that was not me. Probably.",
            "How embarrassing.",
            "I do apologize for the acoustics.",
        ],
        "engineering": [
            "Pressure differential resolved via posterior exhaust port.",
            "Gaseous byproduct successfully vented. No leaks detected.",
            "System emitted an unscheduled low-frequency acoustic event.",
            "Valve actuation complete. Backpressure normalized.",
            "Thermal expansion of intestinal gases within expected tolerance.",
        ],
        "dramatic": [
            "AND LO, THE HEAVENS THEMSELVES DID TREMBLE.",
            "A sound echoed forth, ancient and terrible.",
            "It was, in that moment, the loudest thing in the room.",
            "The walls themselves seemed to flinch.",
            "A legend was born in that single, thunderous moment.",
        ],
        "benchmark": [
            "Trial 1/1: throughput 1 fart/sec, latency negligible.",
            "Benchmark complete. Result: PASS (audible).",
            "p99 fart latency: 4ms. Within acceptable bounds.",
            "Mean acoustic amplitude: acceptable. Variance: concerning.",
            "Regression test passed. No silent failures detected.",
        ],
        "mission_control": [
            "Houston, we have a fart.",
            "Telemetry confirms successful pressure relief.",
            "All systems nominal... except for that one.",
            "Ground control to Major Fart.",
            "The eagle has... well, you know.",
        ],
    }

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def generate(self, style: str = "default") -> str:
        """Return a random string for the given style."""
        if style not in self._STYLES:
            valid = ", ".join(sorted(self._STYLES))
            raise ValueError(f"Unknown style '{style}'. Valid styles: {valid}")
        return self._rng.choice(self._STYLES[style])

    def available_styles(self) -> list[str]:
        return sorted(self._STYLES.keys())


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Fart Generator — because some problems deserve serious engineering."
    )
    parser.add_argument(
        "--style", "-s",
        default="default",
        help="Fart style to generate (default: default)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for deterministic output"
    )
    parser.add_argument(
        "--list-styles",
        action="store_true",
        help="List available styles and exit"
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=1,
        help="Number of farts to generate (default: 1)"
    )

    args = parser.parse_args()
    generator = FartGenerator(seed=args.seed)

    if args.list_styles:
        print("Available styles:")
        for style in generator.available_styles():
            print(f"  - {style}")
        return

    for _ in range(max(1, args.count)):
        print(generator.generate(args.style))


if __name__ == "__main__":
    main()