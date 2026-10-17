"""General camera I/O helpers.

  - ``ingest``      -- ``kiwi import``: copy clips off a mounted camera,
                       named after the unit their serial resolves to
  - ``picker``      -- the ``kiwi import`` page on localhost for choosing clips
  - ``convert``     -- demux an Insta360 ``.insv`` into per-lens ``.mp4`` streams
  - ``imu``         -- extract the IMU telemetry of a recording to CSV
  - ``registry``    -- match a recording to its ``config/cameras/<unit>/`` entry
                       by the serial in the ``.insv`` telemetry
  - ``extrinsics``  -- read the extrinsic calibration artifacts (the cam-IMU
                       camchains)
  - ``kb4``         -- read the KB4 lens-fit artifacts

The artifact readers live here so the writer (``kiwi.calib``) and its
consumers share one parser without importing each other.

Cross-camera time sync (audio and TMCD) lives in :mod:`kiwi.sync`.

The "unit dir -> named artifact" layout (``intrinsics.npz``,
``extrinsic-imu.yaml``, ``av-sync.yaml``, ...) is resolved in one place: add
an artifact's key to ``registry.ARTIFACT_NAMES`` and resolve through
:meth:`registry.CameraUnit.artifact` (which owns the dated-file pointer
convention); do not hardcode ``config/cameras/<unit>/`` filenames elsewhere.

Frame iteration and capture lifetimes are shared through
:func:`kiwi.core.video.iter_frames` and :func:`kiwi.core.video.video_capture`.
"""
