"""KIWI - Kinematic Interface for the Wild.

Off the shelf, into the wild: bimanual + egocentric robot data collection
with three consumer cameras - two Insta360 X5 on printed Chopstick grippers,
one Insta360 GO3 on a hat - and zero other electronics. Recording starts and
stops by voice ("start recording" / "stop recording"); everything else runs
offline through the stage pipeline:

    sync -> convert -> state -> [head] -> [register] -> eef -> [ground]

then ``kiwi viz`` / ``kiwi export`` / ``kiwi replay`` consume the accepted
episode.

Two device configurations, one pipeline:

* **2x X5** - the standard bimanual setup (tabletop workhorse).
* **2x X5 + GO3** - adds the ego stream; the optional post-fusion ``head``
  stage estimates its hat marker from the wrist cameras' rear views.

Package map - the boundary DAG, enforced by ``tests/test_boundaries.py``::

    core <- {calib, track, sync, register, state, eef, ground, perception}
         <- {episode, viz, export, replay, real2sim}
         <- app <- cli

``core`` imports nothing else from ``kiwi``. Components import
``core`` and never each other. ``episode`` (orchestration), ``viz``,
``export``, ``replay`` and ``real2sim`` compose components.
``app`` composes services for the CLI and GUI; lower layers cannot import it.
When two components need the
same piece, promote it into ``core`` instead of adding a cross-component
import.

Each package's docstring carries its interface contract.
"""

__version__ = "0.1.0"
