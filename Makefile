# Makefile — Monte Carlo RT CPU
# Uso: make         → compila
#      make run     → compila e lancia con 1M fotoni
#      make test    → lancia la suite di test
#      make clean   → rimuove binari e output

CXX      = g++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra -lm
TARGET   = mc_rt_cpu

.PHONY: all run test clean

all: $(TARGET)

$(TARGET): main.cpp physics.h compton.h random.h phantom.h output.h
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp
	@echo "✓ Compilazione OK: ./$(TARGET)"

run: $(TARGET)
	./$(TARGET) 1000000 0 42

run_hetero: $(TARGET)
	./$(TARGET) 1000000 1 42

test: $(TARGET)
	python3 tests.py

clean:
	rm -f $(TARGET) *.csv *.bin *.png *.npy
