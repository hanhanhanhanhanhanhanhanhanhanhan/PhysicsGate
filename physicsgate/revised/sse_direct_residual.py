"""SSE benchmark with the Direct prediction added to the inputs of the residual model (Table S16).

    python -m physicsgate.revised.sse_direct_residual [pipeline arguments]
"""

from __future__ import annotations

import runpy
import sys

from physicsgate.revised import sse_residual


def main() -> None:
    sse_residual.install()  # before the pipeline imports the feature names
    runpy.run_module("physicsgate.submitted.sse.pipeline", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    sys.argv[0] = "physicsgate.submitted.sse.pipeline"
    main()
