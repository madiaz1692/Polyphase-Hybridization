# Polyphase Hybridization (PH) - Reviewer Toolkit Makefile
# Target architectures: x86_64 (AVX2, AVX-512), ARM64 (SVE)

CC = gcc
CFLAGS = -O3 -shared -fPIC -fopenmp
SRC_DIR = ../libPH/src/kernel
CORE_DIR = libPH/core

all: x86_avx2 weights

x86_avx2: $(SRC_DIR)/ph_avx2.c
	$(CC) $(CFLAGS) -mavx2 -mfma -o $(CORE_DIR)/libph_x86.so $<

x86_avx512: $(SRC_DIR)/ph_avx512.c
	$(CC) $(CFLAGS) -march=native -mavx512f -mavx512dq -mavx512bw -mavx512vl -o $(CORE_DIR)/libph_avx512.so $<

arm_sve: $(SRC_DIR)/ph_sve.c
	$(CC) $(CFLAGS) -march=armv8-a+sve -o $(CORE_DIR)/libph_sve.so $<

weights:
	python ../libPH/scripts/bake_weights.py

clean:
	rm -f $(CORE_DIR)/*.so $(CORE_DIR)/*.dll
	rm -f ../libPH/weights/*.npy

.PHONY: all weights clean
