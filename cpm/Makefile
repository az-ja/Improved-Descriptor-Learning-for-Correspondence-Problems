CCPP = g++

SSE_FLAG = -march=native -msse2 -DWITH_SSE

CFLAGS = -w -O3 $(SSE_FLAG) -Iinclude -std=c++11 
LDFLAGS = -lopencv_core -lopencv_highgui

SOURCES_CPP := $(shell find . -name '*.cpp')
OBJ := $(SOURCES_CPP:%.cpp=%.o)
HEADERS := $(shell find . -name '*.h')

all: CPM

.cpp.o:  %.cpp %.h
	$(CCPP) -c -o $@ $(CFLAGS) $+

CPM: $(HEADERS) $(OBJ)
	$(CCPP) -o $@ $(OBJ) $(LDFLAGS)

clean:
	rm -f $(OBJ) CPM

