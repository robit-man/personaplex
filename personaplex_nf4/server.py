"""Launch the vendored Moshi server with the packed-NF4 loader installed first."""

from __future__ import annotations

import runpy

from .direct_nf4 import install_direct_nf4_loader


def main() -> None:
    install_direct_nf4_loader()
    runpy.run_module("moshi.server", run_name="__main__")


if __name__ == "__main__":
    main()

