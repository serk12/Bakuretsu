UNAME_S := $(shell uname)

ifeq ($(UNAME_S), Darwin)
    LDFLAGS = -Xlinker -framework,OpenGL -Xlinker -framework,GLUT
else
    LDFLAGS += -L/usr/local/cuda/samples/common/lib/linux/x86_64
    LDFLAGS += -lglut -lGL -lGLU -lGLEW
endif
LDFLAGS += -Wno-deprecated-gpu-targets
NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler "-Wall -Wno-deprecated-declarations" -Wno-deprecated-gpu-targets -std=c++11

all: ./build ./build/bakuretsu.exe

./build/bakuretsu.exe: ./build/main.o ./build/explosion.o ./build/interactions.o ./build/cudaManager.o ./build/shader.o
	$(NVCC) $^ -o $@ $(LDFLAGS)

./build/main.o: ./code/main.cpp ./code/header/explosion.hpp
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

./build/explosion.o: ./code/src/explosion.cpp ./code/header/explosion.hpp ./code/header/interactions.hpp ./code/header/cudaManager.h ./code/header/shader.hpp
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

./build/interactions.o: ./code/src/interactions.cpp ./code/header/interactions.hpp
		$(NVCC) $(NVCC_FLAGS) -c $< -o $@

./build/cudaManager.o: ./code/src/cudaManager.cu ./code/header/cudaManager.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

./build/shader.o: ./code/src/shader.cpp ./code/header/shader.hpp
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

./build:
	mkdir ./build

clean:
	rm -rf ./build/*
