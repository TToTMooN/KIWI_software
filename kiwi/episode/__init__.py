"""Episode model + stage orchestration (a composer: may import components).

Owns:

* the ``episode.yaml`` schema (:mod:`kiwi.episode.manifest`) - per-arm camera /
  calibration / tool config references, the shared ``sync:`` block (see
  :mod:`kiwi.core.clock`), optional extra ``cameras:`` streams, task windows.
  ``EpisodeConfig.load`` parses it, except the optional ``scene:``
  back-reference, which :mod:`kiwi.episode.scene` validates. ``kiwi init``
  writes the file first; afterwards only ``write_sync_block`` (the sync
  stage) and ``write_scene_block`` (``kiwi scene init``) edit it.
* ``kiwi init`` (:mod:`kiwi.episode.init`) - probes the recordings in an
  episode's ``raw/`` (:mod:`kiwi.core.layout` spells the tree) and writes the
  manifest, binding each arm to its camera by the clip's **embedded serial**,
  never by its filename.
* the stage engine behind ``kiwi run`` (:mod:`kiwi.episode.stages`) and its run
  record (:mod:`kiwi.episode.run_record`). Stage order:
  ``sync -> convert -> state -> head -> register -> eef -> ground``;
  state is required by default; head, ground and register are optional.

Contract notes:

* **The manifest is the authority on which clip is which arm.** File and
  directory names are not. Read ``ArmSpec.recording``; never infer from a
  path.
* Extra ``cameras:`` streams contribute exactly one video feature each and
  nothing else - no track, registration, tool state, or intrinsics.
* Stage entry points register in :mod:`kiwi.episode.stages`. Each entry lives
  on the episode side because it returns a ``StageResult``, an episode type;
  a component may not import this package.
"""
