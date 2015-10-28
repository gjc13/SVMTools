cmake_minimum_required(VERSION 3.2)
project(NeighbourSVM)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

FIND_PACKAGE(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

INCLUDE_DIRECTORIES(
  ${OpenCV_INCLUDE_DIRS}
)

set(LIBRARIES
    ${OPENCV_LIBS}
    ${OPENCV_LIBRARIES}
)
set(SOURCE_FILES
    main.cpp
    svm.cpp
    svm.h)
add_executable(NeighbourSVM ${SOURCE_FILES} SVMTools.h SVMTools.cpp)

target_link_libraries(NeighbourSVM
${OpenCV_LIBS}
)
