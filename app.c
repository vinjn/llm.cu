#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

// https://huggingface.co/docs/safetensors/en/index
struct SafeTensor {
    uint64_t header_size;
    uint8_t* header_as_jsonstring;
};

struct TensorField {
    char* data_type; // can be F64, F32, F16, BF16, I64, I32, I16, I8, U8, BOOL
    int* shape;
    int offsets[2];
};

int main(int argc, char* argv[])
{
    FILE* fp = fopen("../data/gpt-2.safetensors", "r");
    if (fp == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    struct SafeTensor safe_tensor = {0};

    // Read the first 10 bytes of the file
    char buffer[10];
    size_t bytesRead = fread(buffer, 1, sizeof(safe_tensor.header_size), fp);
    if (bytesRead < sizeof(safe_tensor.header_size)) {
        printf("Error reading file!\n");
        exit(1);
    }

    return 0;
}