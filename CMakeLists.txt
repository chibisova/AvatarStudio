cmake_minimum_required(VERSION 3.21)

project(AvatarStudio
    VERSION 0.1
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets OpenGLWidgets)

add_executable(AvatarStudio
    src/main.cpp
    src/openglwidget.h src/openglwidget.cpp
    src/objloader.h src/objloader.cpp
    src/camera.h src/camera.cpp
    src/mhridentitydata.h src/mhridentitydata.cpp
    src/hoverable_slider.h src/hoverable_slider.cpp
)

target_link_libraries(AvatarStudio
    PRIVATE
        Qt6::Widgets
        Qt6::OpenGLWidgets
)