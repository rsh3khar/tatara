function(tatara_enable_warnings target)
    target_compile_options(
        ${target}
        PRIVATE
            -Wall
            -Wextra
            -Wconversion
            -Werror
            -Wpedantic
            -Wshadow)
endfunction()
