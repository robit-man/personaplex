"""Build the CUDA extension during Jetson setup, never during a live call."""

from __future__ import annotations

from .direct_nf4 import build_kernel


if __name__ == "__main__":
    build_kernel()

