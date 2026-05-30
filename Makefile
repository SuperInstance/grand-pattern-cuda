NVCC := nvcc
NVCCFLAGS := -O2 -arch=sm_70 -Iinclude -std=c++17
LDFLAGS := -lcudart

SRC := src/grand_pattern_cuda.cu
TEST := tests/test_grand_pattern_cuda.cu

OBJ := build/grand_pattern_cuda.o
TEST_BIN := build/test_grand_pattern_cuda

.PHONY: all test clean dirs

all: dirs $(OBJ)

dirs:
	@mkdir -p build

$(OBJ): $(SRC) include/grand_pattern_cuda.h | dirs
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(TEST_BIN): $(TEST) $(OBJ) | dirs
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -rf build
