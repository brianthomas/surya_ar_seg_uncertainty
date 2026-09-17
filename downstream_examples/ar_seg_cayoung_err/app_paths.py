"""
Import-path bootstrap for the AR-segmentation app.

This app was developed inside the ``surya_workshop`` repository. That repository is going
away, so ``workshop_infrastructure/`` and the ``template/`` app have been vendored into
this one and there is no longer any dependency outside this repository.

Two roots go on the path, both derived from this file's location:

* **this directory** — the app's own ``configs``, ``datasets``, ``metrics``, ``models``
  and ``lightning_modules`` modules, imported by their plain names;
* **the repository root** — ``workshop_infrastructure`` (Surya backbone,
  ``HelioNetCDFDataset``, the config dataclasses, the dataloader builders) and
  ``downstream_examples.template`` (whose Lightning module ``lightning_modules/``
  re-exports).

Importing the module is all there is to it::

    import app_paths  # noqa: F401
"""

from __future__ import annotations

import sys
from pathlib import Path

#: This app's directory — home of configs.py, datasets/, metrics/, models/, ...
APP_DIR = Path(__file__).resolve().parent

#: The repository root — home of workshop_infrastructure/ and downstream_examples/.
REPO_ROOT = APP_DIR.parents[1]


def bootstrap() -> None:
    """Put ``APP_DIR`` and ``REPO_ROOT`` on ``sys.path``. Safe to call repeatedly."""
    if not (REPO_ROOT / "workshop_infrastructure").is_dir():
        raise RuntimeError(
            f"workshop_infrastructure/ not found at {REPO_ROOT}. This app expects to live "
            "two levels below the repository root (downstream_examples/<app>/)."
        )
    for root in (APP_DIR, REPO_ROOT):
        entry = str(root)
        if entry not in sys.path:
            sys.path.insert(0, entry)


bootstrap()
