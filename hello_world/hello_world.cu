#include <stdio.h>
#include <stdlib.h>

__global__ void print_from_gpu(void) {
    printf("Hello World! From thread [%d, %d]  from devices \n", threadIdx.x, blockIdx.x);
}

int main(void) {
    printf("Hello World from Host! \n");
    print_from_gpu<<<1,1>>>();
     cudaDeviceSynchronize();
    return 0;
}
