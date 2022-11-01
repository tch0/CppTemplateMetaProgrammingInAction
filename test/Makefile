# https://github.com/tch0/MyConfigurations/blob/master/MakefileTemplate/CppTemplate2.mk

# Makefile template 2:
# For multiple C++ files in one directory, compile into one executable.

# make debug=yes to compile with -g
# make system=windows for windows system

.PHONY : all run
.PHONY .IGNORE : clean

# add your own include path/library path/link library to CXXFLAGS
CXX = g++
CXXFLAGS += -std=c++20
CXXFLAGS += -Wall -Wextra -Wfatal-errors -pedantic-errors -Wshadow
CXXFLAGS += -I../MetaNN
RM = rm

# final target: add your target here
target = test

# debug
ifeq ($(debug), yes)
CXXFLAGS += -g
else
CXXFLAGS += -O3
CXXFLAGS += -DNDEBUG
endif

# filenames and targets
all_source_files := $(wildcard *.cpp)
all_targets := $(target)

# all targetss
all : $(all_targets)

# compile
$(all_targets) : $(all_source_files)
	$(CXX) $^ -o $@ $(CXXFLAGS)

# run
run : $(all_targets)
	./$(all_targets)

# system: affect how to clean and executable file name
# value: windows/unix
system = unix
ifeq ($(system), windows)
all_targets := $(addsuffix .exe, $(all_targets))
RM := del
endif

# clean
clean :
	-$(RM) $(all_targets)
