#define _CRT_NONSTDC_NO_WARNINGS
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "3rdparty/cJSON/cJSON.h"
#include "3rdparty/mman-win32/mman.h"
#include "3rdparty/getopt-for-windows/getopt.h"
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <io.h>



// https://huggingface.co/docs/safetensors/en/index
struct SafeTensor {
    uint64_t header_size;
    uint8_t* header_ptr;
    uint8_t* rest_of_file;

    cJSON* header_json;
};

struct TensorField {
    char* data_type; // can be F64, F32, F16, BF16, I64, I32, I16, I8, U8, BOOL
    int* shape;
    int offsets[2];
};

char* model = "../data/model.safetensors";
int debug = 0;

int parse(int argc, char* argv[]) {

    // Define long options
    static struct option long_options[] = {
        {"model", required_argument, NULL, 'm'},  // Long option for file path
        {"debug", no_argument, NULL, 'd'}, // Long option for number
        {0, 0, 0, 0}                            // End of options
    };

    int opt;
    int option_index = 0;

    // Parse command-line options
    while ((opt = getopt_long(argc, argv, "m:d", long_options, &option_index)) != -1) {
        switch (opt) {
        case 'm':
            model = optarg; // Get the file path
            break;
        case 'd':
            debug = 1; // Convert the argument to an integer
            break;
        default: /* '?' */
            fprintf(stderr, "Usage: %s -m <model> [-d]\n", argv[0]);
            fprintf(stderr, "       %s --model <model> [--dump]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    return 0;
}

int main(int argc, char* argv[]) {
    if (parse(argc, argv))
        return 1;

    struct SafeTensor safe_tensor = { 0 };

#define _S_IREAD  0x0100 // Read permission, owner

    int mode = _S_IREAD;
    int fd = open(model, O_RDONLY, mode);

    printf("Opening file: %s\n", model);

    struct stat file_stat;
    if (fstat(fd, &file_stat) == -1) {
        perror("Error getting file size");
        close(fd);
        exit(EXIT_FAILURE);
    }

    size_t file_size = file_stat.st_size;

    void* mapped_file = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped_file == MAP_FAILED)
    {
        perror("Error mapping file");
        close(fd);
        exit(EXIT_FAILURE);
    }

    printf("File size: %zu\n", file_size);
    // parsing
    safe_tensor.header_size = *(uint64_t*)mapped_file;
    safe_tensor.header_ptr = (uint8_t*)mapped_file + sizeof(uint64_t);
    safe_tensor.rest_of_file = (uint8_t*)mapped_file + sizeof(uint64_t) + safe_tensor.header_size;

    safe_tensor.header_json = cJSON_ParseWithLength((const char*)safe_tensor.header_ptr, safe_tensor.header_size);
    if (safe_tensor.header_json == NULL) {
        const char* error_ptr = cJSON_GetErrorPtr();
        if (error_ptr != NULL) {
            fprintf(stderr, "Error before: %s\n", error_ptr);
        }
    }

    printf("Header size: %zu\n", safe_tensor.header_size);
    // print first 1000 bytes
    printf("\n\n");
    uint64_t preview_size = safe_tensor.header_size > 1000 ? 1000 : safe_tensor.header_size;
    for (int i = 0; i < preview_size; i++) {
        printf("%c", safe_tensor.header_ptr[i]);
    }
    printf("......\n\n");

    cJSON* tensor_field = cJSON_GetObjectItem(safe_tensor.header_json, "tensor");
    cJSON* data_type = cJSON_GetObjectItem(tensor_field, "data_type");
    cJSON* shape = cJSON_GetObjectItem(tensor_field, "shape");
    cJSON* offsets = cJSON_GetObjectItem(tensor_field, "offsets");


    // Convert the cJSON object to a JSON string
    char* json_string = cJSON_Print(safe_tensor.header_json);
    if (json_string == NULL) {
        fprintf(stderr, "Failed to print JSON object\n");
        cJSON_Delete(safe_tensor.header_json);
        return 1;
    }

    if (debug) {
        // Write the JSON string to a file
        const char* filename = "output.json";
        FILE* file = fopen(filename, "w");
        if (file == NULL) {
            perror("Error opening file");
            free(json_string); // Free the JSON string
            cJSON_Delete(safe_tensor.header_json); // Delete the JSON object
            return 1;
        }

        fprintf(file, "%s\n", json_string); // Write the JSON string to the file
        fclose(file); // Close the file

        printf("JSON has been written to %s\n", filename);

    }
    // Clean up
    free(json_string); // Free the JSON string
    cJSON_Delete(safe_tensor.header_json); // Delete the JSON object


    // Unmap the file and close the file descriptor
    if (munmap(mapped_file, file_size) == -1) {
        perror("Error unmapping file");
    }

    close(fd);

    return 0;
}