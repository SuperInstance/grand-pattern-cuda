//
// main.cu — CUDA kernel test runner
// Grand Pattern Fibonacci Dual-Direction Architecture
//

#include <cstdio>

extern int run_all_tests();

int main() {
    printf("Grand Pattern — CUDA Kernel Tests\n");
    printf("==================================\n\n");

    int failures = run_all_tests();
    return failures;
}
