#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "3rdparty/cJSON/cJSON.h"
#include "3rdparty/mmap-windows/mmap-windows.h"
#include <fcntl.h>

// https://huggingface.co/docs/safetensors/en/index
struct SafeTensor {
    uint64_t header_size;
    uint8_t* header_as_jsonstring;
    uint8_t* rest_of_file;
};

struct TensorField {
    char* data_type; // can be F64, F32, F16, BF16, I64, I32, I16, I8, U8, BOOL
    int* shape;
    int offsets[2];
};

const char* filepath = "../data/gpt-2.safetensors";

int main(int argc, char* argv[])
{
#if 0
    FILE* fp = fopen(filepath, "r");
    if (fp == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    struct SafeTensor safe_tensor = {0};

    // Read the first 10 bytes of the file
    size_t bytesRead = fread(&safe_tensor.header_size, 1, sizeof(safe_tensor.header_size), fp);
    if (bytesRead < sizeof(safe_tensor.header_size)) {
        printf("Error reading file!\n");
        exit(1);
    }
#else

    int fd = open(filepath, O_RDONLY);
    if (fd == -1) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Get the size of the file
    struct stat file_stat;
    if (fstat(fd, &file_stat) == -1) {
        perror("Error getting file size");
        close(fd);
        exit(EXIT_FAILURE);
    }

    size_t file_size = file_stat.st_size;


    void* mapped_file = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped_file == MAP_FAILED) {
        perror("Error mapping file");
        close(fd);
        exit(EXIT_FAILURE);
    }

#endif
    //cJSON* json = cJSON_ParseWithOpts(safe_tensor.header_as_jsonstring, NULL, 1);


    return 0;
}