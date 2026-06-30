cmake_minimum_required(VERSION 3.20)
project(vietnam_lpr_edge VERSION 1.0.0 LANGUAGES CXX)

option(VLPR_BUILD_RUNTIME "Build the ONNX/OpenCV edge runtime" ON)
option(VLPR_BUILD_TESTS "Build dependency-light unit tests" ON)
option(VLPR_WARNINGS_AS_ERRORS "Treat project warnings as errors" OFF)
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path containing ONNX Runtime include/ and lib/")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

function(vlpr_apply_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8)
        if(VLPR_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
        if(VLPR_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
endfunction()

add_library(vlpr_text STATIC src/text.cpp)
target_include_directories(vlpr_text PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
vlpr_apply_warnings(vlpr_text)

if(VLPR_BUILD_RUNTIME)
    find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio)

    if(NOT ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
        set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
    endif()
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS ${ONNXRUNTIME_ROOT}
        PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session)
    find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
        HINTS ${ONNXRUNTIME_ROOT}
        PATH_SUFFIXES lib lib64)
    if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
        message(FATAL_ERROR
            "ONNX Runtime C++ was not found. Set ONNXRUNTIME_ROOT to its extracted SDK directory.")
    endif()

    add_library(vlpr_engine STATIC src/engine.cpp)
    target_include_directories(vlpr_engine
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${ONNXRUNTIME_INCLUDE_DIR}
            ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(vlpr_engine PUBLIC vlpr_text ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARY})
    vlpr_apply_warnings(vlpr_engine)

    add_executable(vlpr_edge src/main.cpp)
    target_include_directories(vlpr_edge PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(vlpr_edge PRIVATE vlpr_engine)
    target_compile_definitions(vlpr_edge PRIVATE VLPR_VERSION="${PROJECT_VERSION}")
    vlpr_apply_warnings(vlpr_edge)

    if(WIN32)
        find_file(ONNXRUNTIME_DLL NAMES onnxruntime.dll
            HINTS ${ONNXRUNTIME_ROOT} PATH_SUFFIXES lib bin)
        get_filename_component(OPENCV_RUNTIME_DIR "${OpenCV_DIR}/../bin" ABSOLUTE)
        file(GLOB OPENCV_RUNTIME_DLLS
            "${OPENCV_RUNTIME_DIR}/opencv_world*.dll"
            "${OPENCV_RUNTIME_DIR}/opencv_videoio_ffmpeg*.dll")
        if(NOT ONNXRUNTIME_DLL)
            message(FATAL_ERROR "onnxruntime.dll was not found under ONNXRUNTIME_ROOT")
        endif()
        add_custom_command(TARGET vlpr_edge POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_DLL}" "$<TARGET_FILE_DIR:vlpr_edge>")
        foreach(OPENCV_RUNTIME_DLL IN LISTS OPENCV_RUNTIME_DLLS)
            add_custom_command(TARGET vlpr_edge POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${OPENCV_RUNTIME_DLL}" "$<TARGET_FILE_DIR:vlpr_edge>")
        endforeach()
        install(FILES ${ONNXRUNTIME_DLL} ${OPENCV_RUNTIME_DLLS} DESTINATION bin)
    elseif(UNIX)
        file(GLOB ONNXRUNTIME_SHARED_LIBRARIES "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so*")
        if(NOT ONNXRUNTIME_SHARED_LIBRARIES)
            message(FATAL_ERROR "ONNX Runtime shared libraries were not found under ONNXRUNTIME_ROOT/lib")
        endif()
        set_target_properties(vlpr_edge PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
        install(FILES ${ONNXRUNTIME_SHARED_LIBRARIES} DESTINATION lib)
    endif()

    install(TARGETS vlpr_edge RUNTIME DESTINATION bin)
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/../models/best_vietnam_lpr.onnx
        ${CMAKE_CURRENT_SOURCE_DIR}/models/PP-OCRv6_rec_small.onnx
        ${CMAKE_CURRENT_SOURCE_DIR}/models/ppocrv6_chars.txt
        ${CMAKE_CURRENT_SOURCE_DIR}/models/model_manifest.json
        DESTINATION share/vietnam-lpr-edge/models)
    install(FILES systemd/vietnam-lpr-edge.service
        DESTINATION share/vietnam-lpr-edge/systemd)
    install(FILES README.md ../THIRD_PARTY_NOTICES.md
        DESTINATION share/doc/vietnam-lpr-edge)
endif()

if(VLPR_BUILD_TESTS)
    include(CTest)
    add_executable(vlpr_text_tests tests/test_text.cpp)
    target_link_libraries(vlpr_text_tests PRIVATE vlpr_text)
    vlpr_apply_warnings(vlpr_text_tests)
    add_test(NAME vlpr_text_tests COMMAND vlpr_text_tests)
endif()

set(CPACK_PACKAGE_NAME "vietnam-lpr-edge")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Headless C++ Vietnamese license plate recognition runtime")
set(CPACK_PACKAGE_VENDOR "Vietnam LPR")
set(CPACK_GENERATOR "TGZ")
include(CPack)
