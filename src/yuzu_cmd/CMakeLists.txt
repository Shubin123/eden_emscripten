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

# Emscripten WebAssembly target
if (EMSCRIPTEN)
    set(EDEN_WEB_ASSET_DIR "${PROJECT_SOURCE_DIR}/../eden_web/web" CACHE PATH
        "Directory receiving generated Eden web artifacts")

    function(configure_eden_web_target target output_name initial_memory)
        target_link_libraries(${target} PRIVATE common core input_common frontend_common video_core)
        target_link_libraries(${target} PRIVATE SDL3::SDL3 Threads::Threads)
        set_target_properties(${target} PROPERTIES OUTPUT_NAME "${output_name}" SUFFIX ".html")
        target_link_options(${target} PRIVATE
            --emit-symbol-map
            -sUSE_PTHREADS=1
            "-sPTHREAD_POOL_SIZE=Math.max(4, Math.min(8, (navigator.hardwareConcurrency || 4) + 1))"
            -sINITIAL_MEMORY=${initial_memory}
            -sSTACK_SIZE=2MB
            -sALLOW_MEMORY_GROWTH=1
            -sMAXIMUM_MEMORY=4GB
            -sASSERTIONS=0
            -sNO_DISABLE_EXCEPTION_CATCHING=1
            "-sEXPORTED_FUNCTIONS=['_eden_init','_eden_load_rom','_eden_step_frame','_eden_run_loop','_eden_shutdown','_eden_get_perf_stats','_eden_set_fast_forward','_eden_get_fast_forward','_eden_is_rom_loaded','_malloc','_free','_emscripten_resize_heap']"
            "-sEXPORTED_RUNTIME_METHODS=['ccall','cwrap','FS','HEAPU8','HEAP32']"
            -sNO_EXIT_RUNTIME=1
        )
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${EDEN_WEB_ASSET_DIR}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE_DIR:${target}>/${output_name}.js"
                "${EDEN_WEB_ASSET_DIR}/${output_name}.js"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE_DIR:${target}>/${output_name}.wasm"
                "${EDEN_WEB_ASSET_DIR}/${output_name}.wasm"
            COMMENT "Synchronizing ${output_name} web runtime artifacts"
        )
    endfunction()

    function(add_eden_web_asset_target target output_name all_target)
        set(sync_commands
            COMMAND ${CMAKE_COMMAND} -E make_directory "${EDEN_WEB_ASSET_DIR}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE_DIR:${target}>/${output_name}.js"
                "${EDEN_WEB_ASSET_DIR}/${output_name}.js"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE_DIR:${target}>/${output_name}.wasm"
                "${EDEN_WEB_ASSET_DIR}/${output_name}.wasm"
        )
        if (${all_target})
            add_custom_target(${target}_assets ALL
                DEPENDS ${target}
                ${sync_commands}
                COMMENT "Ensuring web/ contains current ${output_name} runtime"
            )
        else()
            add_custom_target(${target}_assets
                DEPENDS ${target}
                ${sync_commands}
                COMMENT "Ensuring web/ contains current ${output_name} runtime"
            )
        endif()
    endfunction()

    add_executable(eden_web
        emscripten_main.cpp
        emu_window/emu_window_sdl3.cpp
        emu_window/emu_window_sdl3_null.cpp
        sdl_config.cpp
    )
    target_include_directories(eden_web PRIVATE ${RESOURCES_DIR})
    configure_eden_web_target(eden_web eden 512MB)
    add_eden_web_asset_target(eden_web eden TRUE)
endif()
