"""ORB-SLAM3 subprocess boundary for scene building and localization.

The GPL engine is built by third_party/build_orbslam3.sh. Python never
imports or links it: kiwi.track.orbslam.engine constructs argv and runs a
separate process. Imports stay lazy, so CLI help needs no engine build.
"""
