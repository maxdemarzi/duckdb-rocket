cmake_minimum_required(VERSION 3.5...3.29)

set(TARGET_NAME rocket)
project(${TARGET_NAME})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

include_directories(src/include)

# No vcpkg dependencies on purpose. swan's extension_config.cmake carries a lot of hard-won
# ONNX Runtime and Emscripten platform-detection logic and pulls openssl and highs; none of it
# applies here, because rocket_transform is pure arithmetic over doubles. Every dependency not
# taken is a Windows build failure not had.
set(EXTENSION_SOURCES src/rocket_extension.cpp)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
