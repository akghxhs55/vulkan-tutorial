cmake_minimum_required(VERSION 3.29)
project(vulkan_tutorial)

set(CMAKE_CXX_STANDARD 20)

add_executable(vulkan_tutorial sources/main.cpp)

find_package(glfw3 REQUIRED)
target_link_libraries(vulkan_tutorial glfw)

find_package(glm REQUIRED)
target_link_libraries(vulkan_tutorial glm::glm)

find_package(VulkanLoader REQUIRED)
target_link_libraries(vulkan_tutorial Vulkan::Loader)

find_package(stb REQUIRED)
target_link_libraries(vulkan_tutorial stb::stb)
