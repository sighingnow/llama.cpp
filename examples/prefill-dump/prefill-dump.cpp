#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    gpt_params params;

    if (argc == 1 || argv[1][0] == '-') {
        printf("usage: %s MODEL_PATH [PROMPT OR PROMPT_FILE] [N_GPU_LAYERS] [N_BATCH] [N_PREDICT]\n" , argv[0]);
        return 1 ;
    }

    if (argc >= 2) {
        params.model = argv[1];
    }

    if (argc >= 3) {
        params.prompt = argv[2];
    }

    if (argc >= 4) {
        params.n_gpu_layers = std::atoi(argv[3]);
    }

    if (argc >= 5) {
        params.n_batch = std::atoi(argv[4]);
    }

    if (argc >= 6) {
        params.n_predict = std::atoi(argv[5]);
    }

    if (params.prompt.empty()) {
        params.prompt = "Hello my name is";
    }

    std::string prompt_file_name;
    std::vector<std::string> prompts;
    {
        std::ifstream fp(params.prompt);
        if (!fp.is_open()) {
            prompt_file_name = "prompts";
            prompts.push_back(params.prompt);
        } else {
            prompt_file_name = params.prompt;
            std::string line;
            while (std::getline(fp, line)) {
                if (line.empty() || line[0] == '#') {
                    continue;
                }
                prompts.push_back(line);
            }
        }
    }

    // init LLM
    llama_backend_init();

    // initialize the model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;
    llama_model * model = llama_load_model_from_file(params.model.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // initialize the context
    llama_context_params ctx_params = llama_context_default_params();

    ctx_params.seed  = 1234;
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = params.n_batch;
    ctx_params.n_threads = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads_batch == -1 ? params.n_threads : params.n_threads_batch;

    // use fp16 for kv cache
    ctx_params.type_k = GGML_TYPE_F16;
    ctx_params.type_v = GGML_TYPE_F16;

    llama_context * ctx = llama_new_context_with_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // tokenize the prompt
    const bool add_bos = true;  // default, unconfigurable setting in vLLM.

    size_t num_tokens = 0;
    size_t max_token_list_size = 0;
    std::vector<std::vector<llama_token>> tokens_lists(prompts.size());
    for (size_t i = 0; i < prompts.size(); i++) {
        tokens_lists[i] = ::llama_tokenize(ctx, prompts[i], add_bos);
        num_tokens += tokens_lists[i].size();
        max_token_list_size = std::max(max_token_list_size, tokens_lists[i].size());
    }
    llama_seq_id tokens_lists_size = static_cast<int>(tokens_lists.size());

    const int n_ctx    = llama_n_ctx(ctx);
    const int n_kv_req = num_tokens + tokens_lists.size() * std::max(0, params.n_predict);

    LOG_TEE("\n%s: n_ctx = %d, n_kv_req = %d, n_predict = %d\n", __func__, n_ctx, n_kv_req, params.n_predict);

    // create a llama_batch with size 512
    // we use this object to submit token data for decoding
    llama_batch batch = llama_batch_init(num_tokens, 0, tokens_lists.size());

    // evaluate the initial prompt
    for (llama_seq_id i = 0; i < tokens_lists_size; i++) {
        for (size_t j = 0; j < tokens_lists[i].size(); j++) {
            llama_batch_add(batch, tokens_lists[i][j], j, { i }, false);
        }
        // llama_decode will output logits only for the last token of the prompt
        batch.logits[batch.n_tokens - 1] = true;
    }

    if (llama_decode(ctx, batch) != 0) {
        LOG_TEE("%s: llama_decode() failed\n", __func__);
        return 1;
    }

    // see also:
    // - https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py#L133-L135
    // - https://github.com/ggerganov/llama.cpp/blob/master/convert.py#L565
    size_t repermute_k = llama_model_n_embd_head(model);

    // dump the kv-cache
    std::string prompt_prefix_file_name = prompt_file_name + ".prefix";
    {
        std::ofstream out(prompt_prefix_file_name, std::ios::binary);
        if (!out.is_open()) {
            LOG_TEE("%s: failed to open %s\n", __func__, prompt_prefix_file_name.c_str());
            return 1;
        }
        // file schema:
        // - uint32 (number of requests)
        // - [ uint32 (sequence length),
        //     [ uint32 (token id),
        //       uint32 (num layers),
        //       [ size of tensor, tensor,
        //         size of tensor, tensor
        //       ] * number of layers
        //     ] * number of tokens
        //   ] * number of requests

        out.write(reinterpret_cast<char *>(&tokens_lists_size), 4);
        uint32_t n_cache_elements = llama_model_n_embd_k_gqa(model);
        uint32_t k_tensor_size = ggml_row_size(ctx_params.type_k, n_cache_elements);
        uint32_t v_tensor_size = ggml_row_size(ctx_params.type_v, n_cache_elements);
        uint32_t n_layers = llama_model_n_layer(model);
        uint64_t kv_cache_buffer_size = max_token_list_size
            * n_layers
            * static_cast<uint64_t>(k_tensor_size + v_tensor_size);
        // size_t kv_cache_buffer_size = query_kv_cache_buffer_size(ctx);
        printf("%s: allocate kv_cache buffer size = %zu\n", __func__, kv_cache_buffer_size);
        void *buffer = malloc(kv_cache_buffer_size);

        for (llama_seq_id i = 0; i < tokens_lists_size; ++i) {
            uint32_t sequence_length = static_cast<uint32_t>(tokens_lists[i].size());
            uint64_t current_buffer_size = sequence_length
                * n_layers
                * static_cast<uint64_t>(k_tensor_size + v_tensor_size);
            out.write(reinterpret_cast<char *>(&sequence_length), 4);
            if (export_kv_cache_buffers(ctx, buffer, current_buffer_size,
                                        i, 0, sequence_length, 0, n_layers,
                                        repermute_k) != 0) {
                LOG_TEE("%s: export_kv_cache_buffers() failed\n", __func__);
                return 1;
            }
            uint32_t offset = 0;
            for (uint32_t p = 0; p < sequence_length; ++p) {
                uint32_t token_id = tokens_lists[i][p];
                // printf("%s: seq_id: %u, token_id = %u\n", __func__, i, token_id);
                out.write(reinterpret_cast<char *>(&token_id), 4);
                out.write(reinterpret_cast<char *>(&n_layers), 4);
                for (uint32_t l = 0; l < n_layers; ++l) {
                    // write key
                    out.write(reinterpret_cast<char *>(&k_tensor_size), 4);
                    out.write(reinterpret_cast<char *>(buffer) + offset, k_tensor_size);
                    offset += k_tensor_size;
                    // write value
                    out.write(reinterpret_cast<char *>(&v_tensor_size), 4);
                    out.write(reinterpret_cast<char *>(buffer) + offset, v_tensor_size);
                    offset += v_tensor_size;
                }
            }
            if (offset != current_buffer_size) {
                LOG_TEE("%s: inconsistent: offset = %u, current_buffer_size = %zu\n",
                        __func__, offset, current_buffer_size);
                return 1;
            }
            LOG_TEE("%s: exported: sequence_length = %u, n_layers = %u, current_buffer_size = %zu\n",
                    __func__, sequence_length, n_layers, current_buffer_size);
        }
    }

    llama_print_timings(ctx);

    fprintf(stderr, "\n");

    llama_batch_free(batch);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
