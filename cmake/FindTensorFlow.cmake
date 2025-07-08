# FindTensorFlow.cmake
# Find TensorFlow C API library and headers
#
# This will define the following variables:
#   TensorFlow_FOUND        - True if TensorFlow is found
#   TensorFlow_INCLUDE_DIRS - Include directories for TensorFlow
#   TensorFlow_LIBRARIES    - Libraries to link against
#   TensorFlow_VERSION      - Version of TensorFlow found

# Try to find TensorFlow using pkg-config first
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(PC_TensorFlow QUIET tensorflow)
endif()

# Find the header files
find_path(TensorFlow_INCLUDE_DIR
    NAMES tensorflow/c/c_api.h
    HINTS
        ${PC_TensorFlow_INCLUDE_DIRS}
        $ENV{TENSORFLOW_ROOT}/include
        /usr/local/include
        /usr/include
    PATH_SUFFIXES
        tensorflow
)

# Find the library
find_library(TensorFlow_LIBRARY
    NAMES tensorflow
    HINTS
        ${PC_TensorFlow_LIBRARY_DIRS}
        $ENV{TENSORFLOW_ROOT}/lib
        /usr/local/lib
        /usr/lib
)

# Try to extract version from header if found
if(TensorFlow_INCLUDE_DIR)
    file(READ "${TensorFlow_INCLUDE_DIR}/tensorflow/c/c_api.h" _tensorflow_header)
    string(REGEX MATCH "#define TF_MAJOR_VERSION ([0-9]+)" _match "${_tensorflow_header}")
    if(_match)
        set(TensorFlow_VERSION_MAJOR ${CMAKE_MATCH_1})
    endif()
    string(REGEX MATCH "#define TF_MINOR_VERSION ([0-9]+)" _match "${_tensorflow_header}")
    if(_match)
        set(TensorFlow_VERSION_MINOR ${CMAKE_MATCH_1})
    endif()
    string(REGEX MATCH "#define TF_PATCH_VERSION ([0-9]+)" _match "${_tensorflow_header}")
    if(_match)
        set(TensorFlow_VERSION_PATCH ${CMAKE_MATCH_1})
    endif()
    
    if(TensorFlow_VERSION_MAJOR AND TensorFlow_VERSION_MINOR AND TensorFlow_VERSION_PATCH)
        set(TensorFlow_VERSION "${TensorFlow_VERSION_MAJOR}.${TensorFlow_VERSION_MINOR}.${TensorFlow_VERSION_PATCH}")
    endif()
endif()

# Handle standard find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorFlow
    FOUND_VAR TensorFlow_FOUND
    REQUIRED_VARS TensorFlow_LIBRARY TensorFlow_INCLUDE_DIR
    VERSION_VAR TensorFlow_VERSION
)

# Set the output variables
if(TensorFlow_FOUND)
    set(TensorFlow_LIBRARIES ${TensorFlow_LIBRARY})
    set(TensorFlow_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIR})
    
    # Create imported target
    if(NOT TARGET TensorFlow::TensorFlow)
        add_library(TensorFlow::TensorFlow UNKNOWN IMPORTED)
        set_target_properties(TensorFlow::TensorFlow PROPERTIES
            IMPORTED_LOCATION "${TensorFlow_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorFlow_INCLUDE_DIR}"
        )
    endif()
endif()

# Mark variables as advanced
mark_as_advanced(
    TensorFlow_INCLUDE_DIR
    TensorFlow_LIBRARY
) 