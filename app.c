#define _CRT_NONSTDC_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "3rdparty/cJSON/cJSON.h"
#include "3rdparty/mman-win32/mman.h"
#include "3rdparty/getopt-for-windows/getopt.h"
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <io.h>

// https://huggingface.co/docs/safetensors/en/index

struct MetaData
{
    char* format;
};

struct LN_2
{
    char* name;
    int shape[2];
    int data_offsets[2];
};

struct SafeTensor
{
    uint64_t header_size;
    uint8_t* header_ptr;
    uint8_t* rest_of_file;

    cJSON* header_json;

    struct MetaData metadata;

    struct TensorField* tensor_fields;
};

struct TensorField
{
    char* name;
    char* data_type; // can be F64, F32, F16, BF16, I64, I32, I16, I8, U8, BOOL
    int shapes[4];
    int offsets[2];
};

struct ModelConfig
{
    char* architectures;
    int attention_dropout; // 0.0,
    int bos_token_id; // 151643,
    int eos_token_id; // 151643,
    char* hidden_act; // "silu",
    int hidden_size; // 1536,
    float initializer_range; // 0.02,
    int intermediate_size; // 8960,
    int max_position_embeddings; // 131072,
    int max_window_layers; // 21,
    char* model_type; // "qwen2",
    int num_attention_heads; // 12,
    int num_hidden_layers; // 28,
    int num_key_value_heads; // 2,
    float rms_norm_eps; // 1e-06,
    int rope_theta; // 10000,
    int sliding_window; // 4096,
    int tie_word_embeddings; // false,
    char* torch_dtype; // "bfloat16",
    char* transformers_version; // "4.44.0",
    int use_cache; // true,
    int use_mrope; // false,
    int use_sliding_window; // false,
    int vocab_size; // 151936

};

char* model = "../../DeepSeek-R1-Distill-Qwen-1.5B/model.safetensors";
char* config_json = "../../DeepSeek-R1-Distill-Qwen-1.5B/config.json";
char* tokenizer_json = "../../DeepSeek-R1-Distill-Qwen-1.5B/tokenizer.json";
char* tokenizer_config = "../../DeepSeek-R1-Distill-Qwen-1.5B/tokenizer_config.json";
int debug = 0;

int parse(int argc, char* argv[])
{

    // Define long options
    static struct option long_options[] = {
        {"model", required_argument, NULL, 'm'}, // Long option for file path
        {"config", required_argument, NULL, 'c'}, // Long option for file path
        {"tokenizer", required_argument, NULL, 't'}, // Long option for file path
        {"debug", no_argument, NULL, 'd'},       // Long option for number
        {0, 0, 0, 0}                             // End of options
    };

    int opt;
    int option_index = 0;

    // Parse command-line options
    while ((opt = getopt_long(argc, argv, "m:c:t:d", long_options, &option_index)) != -1)
    {
        switch (opt)
        {
        case 'm':
            model = optarg;
            break;
        case 'c':
            config_json = optarg;
            break;
        case 't':
            tokenizer_json = optarg;
            break;
        case 'd':
            debug = 1;
            break;
        default: /* '?' */
            fprintf(stderr, "Usage: %s -m <model> [-d]\n", argv[0]);
            fprintf(stderr, "       %s --model <model> [--dump]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    return 0;
}

int compare_numeric_strings(const void* lhs, const void* rhs) {
    const char* left = lhs;
    const char* right = rhs;

    // Convert strings to integers for comparison
    int num_a = atoi(left);
    int num_b = atoi(right);

    // Compare the numeric values
    return (num_a > num_b) - (num_a < num_b);
}



int compare_TensorField(const void* lhs, const void* rhs) {
    const struct TensorField* left = lhs;
    const struct TensorField* right = rhs;

    return strcmp(left->name, right->name);
}

// Function to read file contents into a string
char* read_file(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        return NULL;
    }

    fseek(file, 0, SEEK_END);  // Move to end of file
    long length = ftell(file); // Get file size
    fseek(file, 0, SEEK_SET);  // Move back to beginning

    char* json_string = malloc(length + 1);
    if (!json_string) {
        perror("Memory allocation failed");
        fclose(file);
        return NULL;
    }

    fread(json_string, 1, length, file);
    fclose(file);

    json_string[length] = '\0'; // Null-terminate the string
    return json_string;
}

struct Token
{
    char* token;
    int id;
};

struct Token* vocab_tokens;
int vocab_size = 0;

// Add token to vocab
void add_to_vocab(const char* token, int id) {
    vocab_tokens[vocab_size].token = strdup(token);
    vocab_tokens[vocab_size].id = id;
    vocab_size++;
}

// Find token ID
int get_token_id(const char* token) {
    for (int i = 0; i < vocab_size; i++) {
        if (strcmp(vocab_tokens[i].token, token) == 0) {
            return vocab_tokens[i].id;
        }
    }
    return -1; // Unknown token
}

// Perform BPE tokenization
void bpe_tokenize(const char* text) {
    char buffer[256];
    strcpy(buffer, text);

    char* token = strtok(buffer, " ");
    while (token) {
        int id = get_token_id(token);
        if (id != -1) {
            printf("%d ", id);
        }
        else {
            printf("[UNK] ");
        }
        token = strtok(NULL, " ");
    }
    printf("\n");
}



int main(int argc, char* argv[])
{
    if (parse(argc, argv))
        return 1;

    struct ModelConfig model_config = { 0 };

    {
        printf("config_json: %s\n", config_json);
        char* string = read_file(config_json);

        cJSON* json = cJSON_Parse(string);

        model_config.vocab_size = cJSON_GetObjectItem(json, "vocab_size")->valueint;

        // TODO: add more fields

        cJSON_Delete(json);
        free(string);
    }

    vocab_tokens = calloc(model_config.vocab_size, sizeof(struct Token));
    {
        printf("tokenizer_json: %s\n", tokenizer_json);
        char* string = read_file(tokenizer_json);

        cJSON* json = cJSON_Parse(string);

        cJSON* model = cJSON_GetObjectItem(json, "model");
        cJSON* vocab = cJSON_GetObjectItem(model, "vocab");
        cJSON* item = NULL;
        cJSON_ArrayForEach(item, vocab)
        {
           add_to_vocab(item->string, item->valueint);
        }

        cJSON_Delete(json);
        free(string);
    }

    struct SafeTensor safe_tensor = { 0 };

#define _S_IREAD 0x0100 // Read permission, owner

    int mode = _S_IREAD;
    int fd = open(model, O_RDONLY, mode);

    printf("Opening file: %s\n", model);

    struct stat file_stat;
    if (fstat(fd, &file_stat) == -1)
    {
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
    if (safe_tensor.header_json == NULL)
    {
        const char* error_ptr = cJSON_GetErrorPtr();
        if (error_ptr != NULL)
        {
            fprintf(stderr, "Error before: %s\n", error_ptr);
        }
    }

    printf("Header size: %zu\n", safe_tensor.header_size);
    // print first 1000 bytes
    printf("\n\n");
    uint64_t preview_size = safe_tensor.header_size > 1000 ? 1000 : safe_tensor.header_size;
    for (int i = 0; i < preview_size; i++)
    {
        printf("%c", safe_tensor.header_ptr[i]);
    }
    printf("......\n\n");

    int size = cJSON_GetArraySize(safe_tensor.header_json);
    safe_tensor.tensor_fields = calloc(size - 1, sizeof(struct TensorField));

    // TODO: there is an potential off-by-one error here, is __metadata__ always the first item?

    for (int i = 0; i < size; i++)
    {
        cJSON* item = cJSON_GetArrayItem(safe_tensor.header_json, i);
        if (strcmp(item->string, "__metadata__") == 0)
        {
            cJSON* format = cJSON_GetObjectItem(item, "format");
            char* format_value = cJSON_GetStringValue(format);
            printf("__metadata__.Format: %s\n", format_value);
            safe_tensor.metadata.format = format_value;
        }
        else
        {
            struct TensorField* tensor_field = &safe_tensor.tensor_fields[i - 1];

            tensor_field->name = item->string;

            cJSON* dtype = cJSON_GetObjectItem(item, "dtype");
            char* dtype_value = cJSON_GetStringValue(dtype);
            tensor_field->data_type = dtype_value;

            cJSON* shape = cJSON_GetObjectItem(item, "shape");
            cJSON* shape_item = NULL;
            int j = 0;
            cJSON_ArrayForEach(shape_item, shape)
            {
                tensor_field->shapes[j++] = shape_item->valueint;
            }

            cJSON* offset = cJSON_GetObjectItem(item, "data_offsets");
            cJSON* offset_item = NULL;
            j = 0;
            cJSON_ArrayForEach(offset_item, offset)
            {
                tensor_field->offsets[j++] = offset_item->valueint;
            }
        }
    }

    // sort safe_tensor.tensor_fields
    qsort(safe_tensor.tensor_fields, size - 1, sizeof(struct TensorField), compare_TensorField);
    for (int i = 0; i < size - 1; i++)
    {
        printf("TensorField: %s, shape: [%d, %d, %d, %d], offset: [%d, %d]\n", safe_tensor.tensor_fields[i].name,
            safe_tensor.tensor_fields[i].shapes[0], safe_tensor.tensor_fields[i].shapes[1],
            safe_tensor.tensor_fields[i].shapes[2], safe_tensor.tensor_fields[i].shapes[3],
            safe_tensor.tensor_fields[i].offsets[0], safe_tensor.tensor_fields[i].offsets[1]);
    }

    // Convert the cJSON object to a JSON string
    char* json_string = cJSON_Print(safe_tensor.header_json);
    if (json_string == NULL)
    {
        fprintf(stderr, "Failed to print JSON object\n");
        cJSON_Delete(safe_tensor.header_json);
        return 1;
    }

    if (debug)
    {
        // Write the JSON string to a file
        const char* filename = "output.json";
        FILE* file = fopen(filename, "w");
        if (file == NULL)
        {
            perror("Error opening file");
            free(json_string);                     // Free the JSON string
            cJSON_Delete(safe_tensor.header_json); // Delete the JSON object
            return 1;
        }

        fprintf(file, "%s\n", json_string); // Write the JSON string to the file
        fclose(file);                       // Close the file

        printf("JSON has been written to %s\n", filename);
    }
    // Clean up
    free(json_string);                     // Free the JSON string
    cJSON_Delete(safe_tensor.header_json); // Delete the JSON object

    // Unmap the file and close the file descriptor
    if (munmap(mapped_file, file_size) == -1)
    {
        perror("Error unmapping file");
    }

    close(fd);

    return 0;
}