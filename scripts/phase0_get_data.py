#!/usr/bin/env python3
"""Phase 0: fetch the real IEEE-CIS dataset if Kaggle credentials are
available; otherwise the rest of the pipeline transparently falls back to
the synthetic stand-in (see src/sfraud/data/synthetic.py)."""

from sfraud.data.download import main

if __name__ == "__main__":
    main()
