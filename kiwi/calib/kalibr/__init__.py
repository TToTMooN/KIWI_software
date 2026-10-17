"""The Kalibr container seam: staged input, one docker run, distilled output.

Kalibr accepts only ROS1 bags (its CLIs are ``--bag``-required and its reader
layer is built on ``rosbag.Bag`` - there is no folder reader), so the bag has
to exist. The host never builds it: it stages a plain folder
(:mod:`kiwi.calib.kalibr.layout`), and the container's own
``kalibr_bagcreater`` assembles the bag before the solve, inside the same
``docker run``. ROS1 lives entirely behind the Docker boundary; no ROS-shaped
package (``rosbags`` included) ever appears on the host.

Modules:

* ``layout``   - the staged-folder format contract, pinned to bagcreater's
  source line by line; pure stdlib
* ``manifest`` - the ``kiwi_calib.json`` run manifest: facts recorded at
  staging time, never read back out of the bag
* ``staging``  - mp4 -> gray frames, ``.insv`` telemetry -> ``imu0.csv``
* ``yamls``    - the camchain / imu / aprilgrid YAMLs Kalibr consumes
* ``docker``   - the single-run argv builder + log streaming

The image (default ``kalibr:latest``, override ``$KIWI_KALIBR_IMAGE``) is
built once by hand from the submodule's ``Dockerfile_ros1_20_04`` - see
``docs/hardware.md``.
"""
