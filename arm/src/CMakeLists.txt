cmake_minimum_required(VERSION 3.10)
project(ur_impedance_controller)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

link_directories(/usr/local/lib)

# Create a directory for CMake modules if it doesn't exist
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find dependencies
find_package(ur_rtde REQUIRED)
find_package(OpenSSL REQUIRED)

# For ZeroMQ, use pkg-config as a fallback
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

# Eigen3
find_package(Eigen3 REQUIRED)
include_directories(/usr/include/eigen3)

# Find Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Include Python headers
include_directories(${Python3_INCLUDE_DIRS})

# Add executable
add_executable(impedance_controller impedance_controller.cpp)
add_executable(calibration_controller impedance_controller.cpp)
target_compile_definitions(calibration_controller PRIVATE CALIBRATION_ONLY=1)
add_executable(calibration_guard_test calibration_guard_test.cpp)
target_include_directories(calibration_controller PRIVATE ${ZMQ_INCLUDE_DIRS})
target_link_libraries(calibration_controller PRIVATE ur_rtde::rtde ${ZMQ_LIBRARIES} OpenSSL::Crypto)
target_link_libraries(calibration_guard_test PRIVATE OpenSSL::Crypto)
add_executable(state_publisher state_publisher.cpp)

# Create a test executable
add_executable(spline_test spline_test.cpp)


# Include directories
target_include_directories(impedance_controller 
    PRIVATE 
        ${ZMQ_INCLUDE_DIRS}
)

target_include_directories(state_publisher 
    PRIVATE 
        ${ZMQ_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(impedance_controller
    PRIVATE
        ur_rtde::rtde
        ${ZMQ_LIBRARIES}
        OpenSSL::Crypto
)

target_link_libraries(state_publisher
    PRIVATE
        ur_rtde::rtde
        ${ZMQ_LIBRARIES}
)


target_compile_options(state_publisher
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

# Install
install(TARGETS impedance_controller DESTINATION bin)
install(TARGETS state_publisher DESTINATION bin)
