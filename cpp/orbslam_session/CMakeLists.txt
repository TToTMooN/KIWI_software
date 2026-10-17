# kiwi-orbslam-session: the explicit-argument wrapper around ORB-SLAM3.
#
# Built by third_party/build_orbslam3.sh against the STAGED, patched engine
# tree (never the pristine submodule): pass -DORBSLAM_SRC=<staged tree>.
# The binary is a combined GPL-3.0 work - see LICENSE beside this file and
# third_party/NOTICE.
cmake_minimum_required(VERSION 3.16)
project(kiwi_orbslam_session CXX)

if(NOT ORBSLAM_SRC)
  message(FATAL_ERROR "pass -DORBSLAM_SRC=<staged ORB-SLAM3 source tree>")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

add_executable(kiwi-orbslam-session main.cc)
target_include_directories(kiwi-orbslam-session PRIVATE
  ${ORBSLAM_SRC}
  ${ORBSLAM_SRC}/include
  ${ORBSLAM_SRC}/include/CameraModels
  ${ORBSLAM_SRC}/Thirdparty/Sophus
  ${Pangolin_INCLUDE_DIRS}
)
target_link_libraries(kiwi-orbslam-session
  ${ORBSLAM_SRC}/lib/libORB_SLAM3.so
  ${ORBSLAM_SRC}/Thirdparty/DBoW2/lib/libDBoW2.so
  ${ORBSLAM_SRC}/Thirdparty/g2o/lib/libg2o.so
  ${OpenCV_LIBS}
  ${Pangolin_LIBRARIES}
  Eigen3::Eigen
)
# The engine .so lives inside the build tree; bake its location in so the
# wrapper runs without LD_LIBRARY_PATH gymnastics.
set_target_properties(kiwi-orbslam-session PROPERTIES
  BUILD_RPATH "${ORBSLAM_SRC}/lib;${ORBSLAM_SRC}/Thirdparty/DBoW2/lib;${ORBSLAM_SRC}/Thirdparty/g2o/lib"
)
