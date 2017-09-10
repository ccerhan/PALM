cmake_minimum_required(VERSION 3.7)
project(PALM)

set(CMAKE_CXX_STANDARD 11)

# Import OpenCV
find_package(OpenCV)
if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif ()

include_directories(PALM)

set(SOURCES
        PALM/HistogramBuilder.cpp
        PALM/HistogramBuilder.h
        PALM/IlluminationFilter.cpp
        PALM/IlluminationFilter.h
        PALM/PALM.cpp
        PALM/PALM.h
        PALM/PatternImageExtractor.cpp
        PALM/PatternImageExtractor.h
        PALM/ZernikeBaseGenerator.cpp
        PALM/ZernikeBaseGenerator.h
        )

add_executable(PALM main.cpp ${SOURCES})
target_link_libraries(PALM ${OpenCV_LIBS})
