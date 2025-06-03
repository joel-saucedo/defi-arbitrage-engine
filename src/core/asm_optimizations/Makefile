# filepath: src/core/asm_optimizations/Makefile
# ultra-optimized build configuration for assembly mathematical operations
# supports x86_64 with avx2/avx512 extensions

CC = gcc
AS = nasm
AR = ar

# aggressive optimization flags
CFLAGS = -O3 -march=native -mtune=native -flto -ffast-math -funroll-loops \
         -fomit-frame-pointer -finline-functions -fPIC -Wall -Wextra \
         -mavx2 -mfma -msse4.2 -mbmi2 -mlzcnt -mpopcnt

# assembly flags
ASFLAGS = -f elf64 -g -F dwarf

# shared library flags
LDFLAGS = -shared -fPIC -Wl,-O1 -Wl,--as-needed -Wl,--strip-all

# source files
ASM_SOURCES = math_ops.asm
C_SOURCES = asm_interface.c
OBJ_FILES = math_ops.o asm_interface.o

# output library
TARGET = libasm_math.so

# python ctypes test
PYTHON_TEST = test_asm_integration.py

.PHONY: all clean test benchmark install

all: $(TARGET)

$(TARGET): $(OBJ_FILES)
	@echo "[ld] linking optimized shared library..."
	$(CC) $(LDFLAGS) -o $@ $^ -lm
	@echo "[ok] built $(TARGET) with assembly optimizations"

math_ops.o: math_ops.asm
	@echo "[asm] assembling simd mathematical operations..."
	$(AS) $(ASFLAGS) -o $@ $<

asm_interface.o: asm_interface.c
	@echo "[cc] compiling c interface with avx2 optimizations..."
	$(CC) $(CFLAGS) -c -o $@ $<

test: $(TARGET)
	@echo "[test] running python ctypes integration tests..."
	python3 $(PYTHON_TEST)

benchmark: $(TARGET)
	@echo "[bench] performance benchmarking assembly vs c implementations..."
	@python3 $(PYTHON_TEST)

clean:
	@echo "[clean] removing build artifacts..."
	rm -f $(OBJ_FILES) $(TARGET) *.so *.a
	@echo "[ok] clean complete"

install: $(TARGET)
	@echo "[install] copying optimized library to system path..."
	sudo cp $(TARGET) /usr/local/lib/
	sudo ldconfig
	@echo "[ok] library installed and registered"

# development targets
debug: CFLAGS += -g -DDEBUG -O0
debug: ASFLAGS += -g
debug: $(TARGET)

profile: CFLAGS += -pg
profile: $(TARGET)

# architecture-specific optimizations
avx512: CFLAGS += -mavx512f -mavx512dq
avx512: $(TARGET)

# generate assembly listing for inspection
listing: math_ops.asm
	$(AS) $(ASFLAGS) -l math_ops.lst -o math_ops.o $<
	@echo "[ok] assembly listing generated: math_ops.lst"

# static analysis
analyze:
	@echo "[analysis] running static code analysis..."
	cppcheck --enable=all --std=c11 $(C_SOURCES)
	clang-static-analyzer $(C_SOURCES)

# memory usage analysis
valgrind: $(TARGET)
	@echo "[valgrind] memory analysis with optimized binary..."
	valgrind --tool=memcheck --leak-check=full python3 $(PYTHON_TEST)

# performance profiling
perf: $(TARGET)
	@echo "[perf] cpu profiling assembly optimizations..."
	@echo "run: perf record -g python3 $(PYTHON_TEST) && perf report"

# documentation
docs:
	@echo "[docs] generating assembly documentation..."
	@echo "# assembly mathematical operations" > README_ASM.md
	@echo "" >> README_ASM.md
	@echo "## functions" >> README_ASM.md
	@echo "" >> README_ASM.md
	@grep -E "^[a-zA-Z_][a-zA-Z0-9_]*:" math_ops.asm | sed 's/:/:\n/' >> README_ASM.md
	@echo "" >> README_ASM.md
	@echo "## performance characteristics" >> README_ASM.md
	@echo "- avx2 simd vectorization" >> README_ASM.md
	@echo "- zero-allocation operations" >> README_ASM.md 
	@echo "- cache-optimized memory access patterns" >> README_ASM.md
	@echo "- fused multiply-add (fma) instructions" >> README_ASM.md
	@echo "[ok] documentation generated: README_ASM.md"

# compiler flags information
flags:
	@echo "compilation flags:"
	@echo "  CFLAGS: $(CFLAGS)"
	@echo "  ASFLAGS: $(ASFLAGS)" 
	@echo "  LDFLAGS: $(LDFLAGS)"
	@echo ""
	@echo "cpu capabilities:"
	@cat /proc/cpuinfo | grep -E "(avx|sse|fma)" | head -1
