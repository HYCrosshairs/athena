#include "common/utils.h"

void cpu_hello();
void gpu_hello();

int main()
{
    print_hello();
    cpu_hello();
    gpu_hello();
    return 0;
}

