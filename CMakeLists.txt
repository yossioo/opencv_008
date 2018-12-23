cmake_minimum_required(VERSION 3.13)
project(opencv_008)

#set(CMAKE_CXX_STANDARD 17)
add_definitions(-std=c++11)


find_package(OpenCV 3 REQUIRED)
#find_package(PkgConfig)
find_package(X11 REQUIRED)

#pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
#pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)

include_directories(
        include
        # src
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

add_executable(main main.cpp)
target_link_libraries(main
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        X11)

add_executable(new_main new_main.cpp)
target_link_libraries(new_main
        ${X11_LIBRARIES}
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        X11)


add_executable(calib_cameras calib_cameras.cpp)
target_link_libraries(calib_cameras
        ${OpenCV_LIBS})
add_executable(harris harris.cpp)
target_link_libraries(harris
        ${OpenCV_LIBS})
