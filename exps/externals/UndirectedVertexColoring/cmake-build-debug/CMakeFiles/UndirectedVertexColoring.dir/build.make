# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

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
CMAKE_COMMAND = "/Users/AntonioShen/Library/Application Support/JetBrains/Toolbox/apps/CLion/ch-0/211.7142.21/CLion.app/Contents/bin/cmake/mac/bin/cmake"

# The command to remove a file.
RM = "/Users/AntonioShen/Library/Application Support/JetBrains/Toolbox/apps/CLion/ch-0/211.7142.21/CLion.app/Contents/bin/cmake/mac/bin/cmake" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/AntonioShen/MLProjects/SATNet/exps/externals/UndirectedVertexColoring

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/AntonioShen/MLProjects/SATNet/exps/externals/UndirectedVertexColoring/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/UndirectedVertexColoring.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/UndirectedVertexColoring.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/UndirectedVertexColoring.dir/flags.make

CMakeFiles/UndirectedVertexColoring.dir/main.c.o: CMakeFiles/UndirectedVertexColoring.dir/flags.make
CMakeFiles/UndirectedVertexColoring.dir/main.c.o: ../main.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/AntonioShen/MLProjects/SATNet/exps/externals/UndirectedVertexColoring/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/UndirectedVertexColoring.dir/main.c.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/UndirectedVertexColoring.dir/main.c.o -c /Users/AntonioShen/MLProjects/SATNet/exps/externals/UndirectedVertexColoring/main.c

CMakeFiles/UndirectedVertexColoring.dir/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/UndirectedVertexColoring.dir/main.c.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/AntonioShen/MLProjects/SATNet/exps/externals/UndirectedVertexColoring/main.c > CMakeFiles/UndirectedVertexColoring.dir/main.c.i

CMakeFiles/UndirectedVertexColoring.dir/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/UndirectedVertexColoring.dir/main.c.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/AntonioShen/MLProjects/SATNet/exps/externals/UndirectedVertexColoring/main.c -o CMakeFiles/UndirectedVertexColoring.dir/main.c.s

# Object files for target UndirectedVertexColoring
UndirectedVertexColoring_OBJECTS = \
"CMakeFiles/UndirectedVertexColoring.dir/main.c.o"

# External object files for target UndirectedVertexColoring
UndirectedVertexColoring_EXTERNAL_OBJECTS =

UndirectedVertexColoring: CMakeFiles/UndirectedVertexColoring.dir/main.c.o
UndirectedVertexColoring: CMakeFiles/UndirectedVertexColoring.dir/build.make
UndirectedVertexColoring: CMakeFiles/UndirectedVertexColoring.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/AntonioShen/MLProjects/SATNet/exps/externals/UndirectedVertexColoring/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable UndirectedVertexColoring"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/UndirectedVertexColoring.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/UndirectedVertexColoring.dir/build: UndirectedVertexColoring

.PHONY : CMakeFiles/UndirectedVertexColoring.dir/build

CMakeFiles/UndirectedVertexColoring.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/UndirectedVertexColoring.dir/cmake_clean.cmake
.PHONY : CMakeFiles/UndirectedVertexColoring.dir/clean

CMakeFiles/UndirectedVertexColoring.dir/depend:
	cd /Users/AntonioShen/MLProjects/SATNet/exps/externals/UndirectedVertexColoring/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/AntonioShen/MLProjects/SATNet/exps/externals/UndirectedVertexColoring /Users/AntonioShen/MLProjects/SATNet/exps/externals/UndirectedVertexColoring /Users/AntonioShen/MLProjects/SATNet/exps/externals/UndirectedVertexColoring/cmake-build-debug /Users/AntonioShen/MLProjects/SATNet/exps/externals/UndirectedVertexColoring/cmake-build-debug /Users/AntonioShen/MLProjects/SATNet/exps/externals/UndirectedVertexColoring/cmake-build-debug/CMakeFiles/UndirectedVertexColoring.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/UndirectedVertexColoring.dir/depend
