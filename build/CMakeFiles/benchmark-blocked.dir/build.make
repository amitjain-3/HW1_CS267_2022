# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /global/common/cori_cle7/software/cmake/3.21.3/bin/cmake

# The command to remove a file.
RM = /global/common/cori_cle7/software/cmake/3.21.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /global/homes/a/amitjain/HW1_CS267_2022

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /global/homes/a/amitjain/HW1_CS267_2022/build

# Include any dependencies generated for this target.
include CMakeFiles/benchmark-blocked.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/benchmark-blocked.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/benchmark-blocked.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/benchmark-blocked.dir/flags.make

CMakeFiles/benchmark-blocked.dir/dgemm-blocked.c.o: CMakeFiles/benchmark-blocked.dir/flags.make
CMakeFiles/benchmark-blocked.dir/dgemm-blocked.c.o: ../dgemm-blocked.c
CMakeFiles/benchmark-blocked.dir/dgemm-blocked.c.o: CMakeFiles/benchmark-blocked.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/homes/a/amitjain/HW1_CS267_2022/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/benchmark-blocked.dir/dgemm-blocked.c.o"
	/opt/cray/pe/craype/2.6.2/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/benchmark-blocked.dir/dgemm-blocked.c.o -MF CMakeFiles/benchmark-blocked.dir/dgemm-blocked.c.o.d -o CMakeFiles/benchmark-blocked.dir/dgemm-blocked.c.o -c /global/homes/a/amitjain/HW1_CS267_2022/dgemm-blocked.c

CMakeFiles/benchmark-blocked.dir/dgemm-blocked.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/benchmark-blocked.dir/dgemm-blocked.c.i"
	/opt/cray/pe/craype/2.6.2/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /global/homes/a/amitjain/HW1_CS267_2022/dgemm-blocked.c > CMakeFiles/benchmark-blocked.dir/dgemm-blocked.c.i

CMakeFiles/benchmark-blocked.dir/dgemm-blocked.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/benchmark-blocked.dir/dgemm-blocked.c.s"
	/opt/cray/pe/craype/2.6.2/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /global/homes/a/amitjain/HW1_CS267_2022/dgemm-blocked.c -o CMakeFiles/benchmark-blocked.dir/dgemm-blocked.c.s

# Object files for target benchmark-blocked
benchmark__blocked_OBJECTS = \
"CMakeFiles/benchmark-blocked.dir/dgemm-blocked.c.o"

# External object files for target benchmark-blocked
benchmark__blocked_EXTERNAL_OBJECTS = \
"/global/homes/a/amitjain/HW1_CS267_2022/build/CMakeFiles/benchmark.dir/benchmark.cpp.o"

benchmark-blocked: CMakeFiles/benchmark-blocked.dir/dgemm-blocked.c.o
benchmark-blocked: CMakeFiles/benchmark.dir/benchmark.cpp.o
benchmark-blocked: CMakeFiles/benchmark-blocked.dir/build.make
benchmark-blocked: CMakeFiles/benchmark-blocked.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/global/homes/a/amitjain/HW1_CS267_2022/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable benchmark-blocked"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/benchmark-blocked.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/benchmark-blocked.dir/build: benchmark-blocked
.PHONY : CMakeFiles/benchmark-blocked.dir/build

CMakeFiles/benchmark-blocked.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/benchmark-blocked.dir/cmake_clean.cmake
.PHONY : CMakeFiles/benchmark-blocked.dir/clean

CMakeFiles/benchmark-blocked.dir/depend:
	cd /global/homes/a/amitjain/HW1_CS267_2022/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /global/homes/a/amitjain/HW1_CS267_2022 /global/homes/a/amitjain/HW1_CS267_2022 /global/homes/a/amitjain/HW1_CS267_2022/build /global/homes/a/amitjain/HW1_CS267_2022/build /global/homes/a/amitjain/HW1_CS267_2022/build/CMakeFiles/benchmark-blocked.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/benchmark-blocked.dir/depend

