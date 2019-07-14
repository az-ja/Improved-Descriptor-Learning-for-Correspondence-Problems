CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
SET(PROJ_NAME CPM)
PROJECT (${PROJ_NAME})

ADD_DEFINITIONS(-DWITH_SSE)

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++14 -O2 -D_GLIBCXX_PARALLEL -fopenmp -L /home/azin/tensorflow/tensorflow/contrib/makefile/gen/lib -ltensorflow-core")

INCLUDE_DIRECTORIES(
./
./include
/home/azin/opencv/include
/home/azin/opencv/include/opencv
/home/azin/opencv/include/opencv2
/home/azin/tensorflow
/home/azin/tensorflow/tensorflow/contrib/makefile/downloads/eigen
/home/azin/tensorflow/tensorflow/contrib/makefile/gen/proto
/home/azin/tensorflow/tensorflow/contrib/makefile/gen/protobuf/include
/home/azin/tensorflow/bazel-genfiles
)

LINK_DIRECTORIES(
/home/azin/opencv/build/lib
/usr/local/lib
/home/azin/tensorflow/bazel-bin/tensorflow
)

ADD_EXECUTABLE(${PROJ_NAME} main.cpp CPM.cpp)

TARGET_LINK_LIBRARIES(${PROJ_NAME}
debug libtensorflow_cc.so
debug libopencv_core.so
debug libopencv_highgui.so
debug libopencv_imgcodecs.so
debug libopencv_imgproc.so
debug libopencv_xfeatures2d.so
optimized libtensorflow_cc.so
optimized libopencv_core.so
optimized libopencv_highgui.so
optimized libopencv_imgcodecs.so
optimized libopencv_imgproc.so
optimized libopencv_xfeatures2d.so)
