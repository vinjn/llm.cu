# Weights

- gpt2 https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors
- DeepSeek-R1-Distill-Qwen-1.5B https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- DeepSeek-R1-Distill-Qwen-7B https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

# TODO

- [ ] Parse all the layers
- [ ] Tokenizer
- [ ] matmul in C and CUDA
- [ ] MLP in C and CUDA
- [ ] attention layer in C and CUDA


# Reference

## Overview
- https://jalammar.github.io/illustrated-transformer/
- https://dugas.ch/artificial_curiosity/GPT_architecture.html
- [Andrej Karpathy - Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)

## Inferencing
- https://www.omrimallis.com/posts/understanding-how-llm-inference-works-with-llama-cpp/

## Tokenizer

- [Andrej Karpathy - Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- [LLM Tokenizers Explained: BPE Encoding, WordPiece and SentencePiece](https://www.youtube.com/watch?v=hL4ZnAWSyuU)
- https://github.com/openai/tiktoken

