cmake_minimum_required(VERSION 3.8)
project(wlr_gazebo)

find_package(ament_cmake REQUIRED)

# ---------- FiveBarClosure Gazebo system plugin ----------
find_package(gz-sim8 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)

add_library(FiveBarClosure SHARED src/FiveBarClosure.cc)
target_link_libraries(FiveBarClosure
  gz-sim8::gz-sim8
  gz-plugin2::register
)
# Gazebo plugin discovery
install(TARGETS FiveBarClosure
  DESTINATION lib/${PROJECT_NAME}
)

# ---------- Launch / world files ----------
install(DIRECTORY launch worlds
  DESTINATION share/${PROJECT_NAME}
)

# ---------- Env hooks so Gazebo finds our plugins ----------
ament_environment_hooks(
  "${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/wlr_gazebo.dsv.in"
)

ament_package()
