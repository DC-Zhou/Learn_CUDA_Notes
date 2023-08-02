CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=global_memory shared_memory shared_memory_02 unifined_memory unifined_memory_initialized unifined_memory_64align unifined_memory_prefetch
BUILD_TYPE=Debug

INCLUDES = -I${CUDA_PATH}/samples/common/inc
NVCC_FLAGS=-m64 -lineinfo #--resource-usage

IS_CUDA_11:=${shell expr `$(NVCC) --version | grep compilation | grep -Eo -m 1 '[0-9]+.[0-9]' | head -1` \>= 11.0}

# Gencode argumentes
SMS = 87
ifeq "$(IS_CUDA_11)" "1"
SMS = 87
endif
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

LIBRARIES += -lgomp
ALL_CCFLAGS += -std=c++11 -Xcompiler -fopenmp -rdc=true $(NVCC_FLAGS) $(INCLUDES)

all : ${TARGET}
	
global_memory: global_memory.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@.out $+ 

shared_memory: shared_memory.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@.out $+ 

shared_memory_02: shared_memory_02.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@.out $+

unifined_memory: unifined_memory.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@.out $+

unifined_memory_initialized: unifined_memory_initialized.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@.out $+

unifined_memory_64align: unifined_memory_64align.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@.out $+

unifined_memory_prefetch: unifined_memory_prefetch.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@.out $+

clean:
	rm -f *.out *.sqlite *.nsys-rep