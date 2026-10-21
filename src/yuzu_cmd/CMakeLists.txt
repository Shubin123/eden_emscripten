# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# SPDX-FileCopyrightText: 2018 yuzu Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

# Credits to Samantas5855 and others for this function.
function(create_resource file output filename)
    # Read hex data from file
    file(READ ${file} filedata HEX)
    # Convert hex data for C compatibility
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," filedata ${filedata})
    # Write data to output file
    set(RESOURCES_DIR "${PROJECT_BINARY_DIR}/dist" PARENT_SCOPE)
    file(WRITE "${PROJECT_BINARY_DIR}/dist/${output}" "const unsigned char ${filename}[] = {${filedata}};\nconst unsigned ${filename}_size = sizeof(${filename});\n")
endfunction()

if (ENABLE_OPENGL)
    list(APPEND OPENGL_SOURCES
        emu_window/emu_window_sdl3_gl.cpp
        emu_window/emu_window_sdl3_gl.h
    )
else()
    set(OPENGL_SOURCES "")
endif()

add_executable(yuzu-cmd
    emu_window/emu_window_sdl3.cpp
    emu_window/emu_window_sdl3.h
    emu_window/emu_window_sdl3_null.cpp
    emu_window/emu_window_sdl3_null.h
    emu_window/emu_window_sdl3_vk.cpp
    emu_window/emu_window_sdl3_vk.h
    sdl_config.cpp
    sdl_config.h
    yuzu.cpp
    yuzu.rc
    ${OPENGL_SOURCES}
)

target_link_libraries(yuzu-cmd PRIVATE common core input_common frontend_common video_core)
if (ENABLE_OPENGL)
    target_link_libraries(yuzu-cmd PRIVATE glad)
endif()
target_link_libraries(yuzu-cmd PRIVATE ${PLATFORM_LIBRARIES} Threads::Threads)

create_resource("../../dist/eden.bmp" "yuzu_cmd/yuzu_icon.h" "yuzu_icon")
target_include_directories(yuzu-cmd PRIVATE ${RESOURCES_DIR})

target_link_libraries(yuzu-cmd PRIVATE SDL3::SDL3)

if(UNIX AND NOT APPLE)
    install(TARGETS yuzu-cmd)
endif()

if(WIN32)
    # compile as a win32 gui application instead of a console application
    if(MSVC)
        set_target_properties(yuzu-cmd PROPERTIES LINK_FLAGS_RELEASE "/SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
    elseif(MINGW)
        set_target_properties(yuzu-cmd PROPERTIES LINK_FLAGS_RELEASE "-Wl,--subsystem,windows")
    endif()
endif()

create_target_directory_groups(yuzu-cmd)

# needed for vma
if (NOT MSVC)
    target_compile_options(yuzu-cmd PRIVATE
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-conversion>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unused-variable>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unused-parameter>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-missing-field-initializers>)
endif()

# Emscripten WebAssembly targets
if (EMSCRIPTEN)
    # Flags shared by the browser and Node builds. main() runs on a pthread so that the core may
    # block; guest fibers are also pthreads, so the pool may grow on demand.
    set(EDEN_WEB_LINK_OPTIONS
        -sPROXY_TO_PTHREAD=1
        -sPTHREAD_POOL_SIZE=48
        -sALLOW_MEMORY_GROWTH=1
        -sALLOW_TABLE_GROWTH=1
        -sMAXIMUM_MEMORY=16GB
        -sSTACK_SIZE=4MB
        -sDEFAULT_PTHREAD_STACK_SIZE=2MB
        -sASSERTIONS=0
        -sEXCEPTION_STACK_TRACES=1
        --emit-symbol-map
    )

    # Build details reported by eden_get_build_info(); the commit itself comes from scm_rev.
    option(EDEN_WEB_DEV "Development build linked without whole-program optimization" OFF)
    find_package(Git QUIET)
    run_git_command(EDEN_WEB_GIT_STATUS status --porcelain --untracked-files=no)
    if ("${EDEN_WEB_GIT_STATUS}" MATCHES "NOTFOUND")
        set(EDEN_WEB_GIT_DIRTY -1)
    elseif ("${EDEN_WEB_GIT_STATUS}" STREQUAL "")
        set(EDEN_WEB_GIT_DIRTY 0)
    else()
        set(EDEN_WEB_GIT_DIRTY 1)
    endif()
    set(EDEN_WEB_DEFINITIONS
        EDEN_WEB_GIT_DIRTY=${EDEN_WEB_GIT_DIRTY}
        EDEN_WEB_DEV_BUILD=$<BOOL:${EDEN_WEB_DEV}>
        EDEN_WEB_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    )

    option(EDEN_WEB_ALLOC_DIAGNOSTICS "Log browser heap growth and texture allocation failures" OFF)
    if (EDEN_WEB_ALLOC_DIAGNOSTICS)
        list(APPEND EDEN_WEB_LINK_OPTIONS "-Wl,--wrap=malloc"
            "SHELL:--pre-js \"${CMAKE_CURRENT_SOURCE_DIR}/emscripten_alloc_pre.js\"")
        list(APPEND EDEN_WEB_DEFINITIONS EDEN_WEB_ALLOC_DIAGNOSTICS)
    endif()

    add_executable(eden_web emscripten_main.cpp emscripten_rom.cpp)
    target_compile_definitions(eden_web PRIVATE ${EDEN_WEB_DEFINITIONS})
    target_link_libraries(eden_web PRIVATE common core input_common frontend_common video_core Threads::Threads)
    set_target_properties(eden_web PROPERTIES OUTPUT_NAME "eden")
    target_link_options(eden_web PRIVATE
        ${EDEN_WEB_LINK_OPTIONS}
        -sENVIRONMENT=web,worker
        -sMODULARIZE=1
        -sEXPORT_NAME=createEdenModule
        -sEXIT_RUNTIME=0
        "-sEXPORTED_FUNCTIONS=['_main','_malloc','_free']"
        "-sEXPORTED_RUNTIME_METHODS=['ccall','cwrap','FS','ENV','UTF8ToString','HEAPF64']"
    )

    add_executable(eden_headless emscripten_main.cpp emscripten_rom.cpp)
    target_link_libraries(eden_headless PRIVATE common core input_common frontend_common video_core Threads::Threads)
    target_compile_definitions(eden_headless PRIVATE EDEN_WEB_HEADLESS ${EDEN_WEB_DEFINITIONS})
    target_link_options(eden_headless PRIVATE
        ${EDEN_WEB_LINK_OPTIONS}
        -sENVIRONMENT=node
        -sNODERAWFS=1
        -sEXIT_RUNTIME=1
        --profiling-funcs
        --pre-js "${CMAKE_CURRENT_SOURCE_DIR}/emscripten_headless_pre.js"
    )
    set_property(TARGET eden_headless APPEND PROPERTY LINK_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/emscripten_headless_pre.js")
    if (EDEN_WEB_ALLOC_DIAGNOSTICS)
        set_property(TARGET eden_web eden_headless APPEND PROPERTY LINK_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/emscripten_alloc_pre.js")
    endif()
endif()
