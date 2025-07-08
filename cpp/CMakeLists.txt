# C++ Trading Execution Layer

# Create library for common components
add_library(trading_core
    src/model_runner.cpp
    src/risk_guard.cpp
    src/api_client.cpp
)

# Link libraries to trading_core
target_link_libraries(trading_core
    ${TensorFlow_LIBRARIES}
    Boost::system
    Threads::Threads
)

# Set include directories for the library
target_include_directories(trading_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TensorFlow_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

# Create the main executable
add_executable(bts_algo_main
    src/main.cpp
)

# Link the main executable with the trading core library
target_link_libraries(bts_algo_main
    trading_core
    ${TensorFlow_LIBRARIES}
    Boost::system
    Threads::Threads
)

# Set target properties
set_target_properties(trading_core PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

set_target_properties(bts_algo_main PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "bts_algo"
)

# Install targets
install(TARGETS bts_algo_main
    RUNTIME DESTINATION bin
)

install(TARGETS trading_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
) 