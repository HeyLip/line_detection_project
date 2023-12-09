cmake_minimum_required(VERSION 3.10)

project(line_detection_project)

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS},
    ${EIGEN3_INCLUDE_DIR}
)

add_executable(${PROJECT_NAME} src/sliding_window.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})