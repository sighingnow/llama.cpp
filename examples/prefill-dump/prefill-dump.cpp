#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static std::tuple<size_t /* buffer_nbytes */,
                  uint32_t /* num_layers */,
                  uint32_t /* tensor_nbytes */> export_kv_cache_buffers(
    llama_context_params ctx_params,
    llama_context * ctx,
    llama_model *model,
    llama_pos token_start_pos,
    llama_pos token_stop_pos,
    int layer0,
    int layer1,
    llama_seq_id seq_id,
    std::vector<uint8_t> &buffer)
{
    // buffer schema: [[(k tensor, v tensor) * layers] * sequence_length]
    //
    // see also:
    // - https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py#L133-L135
    // - https://github.com/ggerganov/llama.cpp/blob/master/convert.py#L565
    size_t repermute_k = llama_model_n_embd_head(model);
    llama_pos sequence_length = token_stop_pos - token_start_pos;

    if (layer0 >= layer1) {
        LOG_TEE("%s: error: invalid layer range: [%d, %d)\n", __func__, layer0, layer1);
        return std::make_tuple(0, 0, 0);
    }

    uint32_t k_cache_elements = llama_model_n_embd_k_gqa(model);
    uint32_t v_cache_elements = llama_model_n_embd_v_gqa(model);
    uint32_t k_tensor_nbytes = ggml_row_size(ctx_params.type_k, k_cache_elements);
    uint32_t v_tensor_nbytes = ggml_row_size(ctx_params.type_v, v_cache_elements);
    uint32_t n_layers = llama_model_n_layer(model);
    uint64_t buffer_nbytes = sequence_length
            * n_layers
            * static_cast<uint64_t>(k_tensor_nbytes + v_tensor_nbytes);
    if (buffer.size() < buffer_nbytes) {
        buffer.resize(buffer_nbytes);
    }

    if (export_kv_cache_buffers(ctx, buffer.data(), buffer_nbytes,
                                seq_id, token_start_pos, token_stop_pos, 0, n_layers,
                                repermute_k) != 0) {
        LOG_TEE("%s: export_kv_cache_buffers() failed\n", __func__);
        return std::make_tuple(0, 0, 0);
    }
    LOG_TEE("%s: exported: sequence_length = %u (%u -> %u), n_layers = %d (%d -> %d), current_buffer_size = %zu\n",
            __func__, sequence_length, token_start_pos, token_stop_pos,
            layer1 - layer0, layer0, layer1, buffer_nbytes);
    return std::make_tuple(buffer_nbytes, layer1 - layer0, k_tensor_nbytes);
}

static int tokenize(
    llama_context * ctx,
    const std::vector<std::string> &prompts,
    std::vector<std::vector<llama_token>> &tokens_lists
) {
    // tokenize the prompt
    const bool add_bos = true;  // default, unconfigurable setting in vLLM.

    size_t max_token_list_size = 0;
    tokens_lists.resize(prompts.size());
    for (size_t i = 0; i < prompts.size(); i++) {
        tokens_lists[i] = ::llama_tokenize(ctx, prompts[i], add_bos);
        max_token_list_size = std::max(max_token_list_size, tokens_lists[i].size());
    }
    return max_token_list_size;
}

static int decode(
    llama_context * ctx,
    llama_sampling_context * ctx_sampling,
    llama_model *model,
    llama_batch &batch,
    llama_token *batch_tokens,
    float *batch_embd,
    int32_t n_batch,
    uint32_t n_embd,
    llama_seq_id seq_id,
    const std::vector<llama_token> &prompt_tokens,  // for hidden state
    llama_pos prompt_start_pos,
    llama_pos prompt_stop_pos,
    const std::vector<std::vector<uint8_t>> &hidden_states,
    llama_pos hidden_state_size,  // if any, should be the same length with [start_pos, stop_pos]
    const std::vector<llama_token> &candidate_tokens,  // for sampling
    std::vector<llama_token> &out_tokens
) {
    llama_pos n_tokens = prompt_tokens.size();
    llama_pos n_candidate_tokens = candidate_tokens.size();

    // LOG_TEE("%s: n_tokens = %d, prompt_start_pos = %d\n", __func__, n_tokens, prompt_start_pos);
    if ((prompt_stop_pos - prompt_start_pos) != n_tokens) {
        LOG_TEE("%s: error: invalid token range: [%d, %d) vs. n_tokens: %d\n",
                __func__, prompt_start_pos, prompt_stop_pos, n_tokens);
        return 1;
    }
    if (n_tokens > n_batch) {
        LOG_TEE("%s: error: too many tokens: %d, mostly allowed (nbatch): %d\n",
                __func__, n_tokens, n_batch);
        return 1;
    }

    llama_batch_clear(batch);

    // set the tokens buffer and unset the batch embd
    batch.token = batch_tokens;
    batch.embd = nullptr;

    for (llama_pos i = 0; i < n_tokens; i++) {
        llama_batch_add(batch,
                        prompt_tokens[i],
                        i + prompt_start_pos,
                        { seq_id },
                        false);
    }

    // When batch.embd exists, llama.cpp will respect the embd buffer
    // as the inputs and skip fetching embd vectors from the model.
    //
    // It is used to feed the hidden states from the server side for
    // last k layers when collaborative decoding is enabled.
    //
    // The `llama_decode_internal()` asserts batch.token and batch.embd
    // cannot exist simultaneously.

    if (hidden_state_size > 0) {
        if (hidden_state_size != n_tokens) {
            LOG_TEE("%s: error: hidden_state_size != n_tokens: %d != %d\n",
                    __func__, hidden_state_size, n_tokens);
            return 1;
        }

        // unset the tokens buffer and allocate embd buffer
        batch.token = nullptr;
        batch.embd = batch_embd;

        // copy the hidden states to the embd buffer
        for (llama_pos i = 0; i < hidden_state_size; i++) {
            memcpy(batch.embd + i * n_embd, hidden_states[i].data(), sizeof(float) * n_embd);
        }
    }

    // keep logits for candidate tokens
    for (llama_pos i = 0; i < n_candidate_tokens; i++) {
        batch.logits[batch.n_tokens - 1 - i] = true;
    }

    // evaluate the current batch with the transformer model
    if (llama_decode(ctx, batch)) {
        LOG_TEE("%s : failed to llama_decode, return code %d\n", __func__, 1);
        return 1;
    }

    out_tokens.clear();
    llama_pos decoding_offset = batch.n_tokens - n_candidate_tokens;
    for (llama_pos idx = 0; idx < n_candidate_tokens; ++idx) {
        llama_pos token_idx = decoding_offset + idx;
        const llama_token id = llama_sampling_sample(ctx_sampling, ctx, nullptr, token_idx);

        llama_sampling_accept(ctx_sampling, ctx, id, true);

        out_tokens.push_back(id);
        if (id == llama_token_eos(model)) {
            break;
        }

        // speculative decoding: mismatched
        if (idx < n_candidate_tokens - 1 && id != candidate_tokens[idx]) {
            break;
        }
    }
    return 0;
}

int main(int argc, char ** argv) {
    gpt_params params;

    llama_pos n_prompt_chunk = 48;

    if (argc == 1 || argv[1][0] == '-') {
        printf("usage: %s MODEL_PATH [PROMPT OR PROMPT_FILE] [N_PROMPT_CHUNK] [N_GPU_LAYERS] [N_SKIP_LAYERS] [N_BATCH] [N_PREDICT]\n" , argv[0]);
        return 1 ;
    }

    if (argc >= 2) {
        params.model = argv[1];
    }

    if (argc >= 3) {
        params.prompt = argv[2];
    }

    if (argc >= 4) {
        n_prompt_chunk = std::atoi(argv[3]);
    }

    if (argc >= 5) {
        params.n_gpu_layers = std::atoi(argv[4]);
    }

    if (argc >= 6) {
        params.n_skip_layers = std::atoi(argv[5]);
    }

    if (argc >= 7) {
        params.n_batch = std::atoi(argv[6]);
    }

    if (argc >= 8) {
        params.n_predict = std::atoi(argv[7]);
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
    uint32_t n_ctx = llama_n_ctx(ctx);
    uint32_t n_embd = llama_model_n_embd(model);
    uint32_t n_layers = llama_model_n_layer(model);

    // sampling context
    params.sparams.temp = 0.0f;  // greedy
    struct llama_sampling_context * ctx_sampling = llama_sampling_init(params.sparams);

    // tokenize the prompt
    std::vector<std::vector<llama_token>> tokens_lists(prompts.size());
    int max_token_list_size = tokenize(ctx, prompts, tokens_lists);
    llama_seq_id tokens_lists_size = tokens_lists.size();

    // create a llama_batch with size 512
    // we use this object to submit token data for decoding
    llama_batch batch = llama_batch_init(params.n_batch, 0, prompts.size());
    llama_token *batch_tokens = batch.token;
    float *batch_embd = (float *) malloc(sizeof(float) * params.n_batch * n_embd);

    std::vector<uint8_t> prefix_caches;

    std::vector<std::vector<uint8_t>> hidden_states;
    std::vector<llama_token> out_tokens;

    // dump the kv-cache
    //
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
    std::string prompt_prefix_file_name = prompt_file_name + ".prefix";
    std::ofstream out(prompt_prefix_file_name, std::ios::binary);
    if (!out.is_open()) {
        LOG_TEE("%s: failed to open %s\n", __func__, prompt_prefix_file_name.c_str());
        return 1;
    }
    out.write(reinterpret_cast<char *>(&tokens_lists_size), 4);

    uint32_t k_cache_elements = llama_model_n_embd_k_gqa(model);
    uint32_t v_cache_elements = llama_model_n_embd_k_gqa(model);
    uint32_t k_tensor_size = ggml_row_size(ctx_params.type_k, k_cache_elements);
    uint32_t v_tensor_size = ggml_row_size(ctx_params.type_v, v_cache_elements);

    for (llama_seq_id seq_id = 0; seq_id < tokens_lists_size; seq_id++) {
        llama_pos n_tokens = tokens_lists[seq_id].size();
        llama_pos n_prefill = n_prompt_chunk * std::max(
            0, (n_tokens + n_prompt_chunk - 1) / n_prompt_chunk - 2);
        out.write(reinterpret_cast<char *>(&n_prefill), 4);

        if (n_prefill == 0) {
            LOG_TEE("%s: exported seqeunce %u: n_tokens = %u\n", __func__, seq_id, n_prefill);
            continue;
        }

        std::vector<llama_token> tokens(tokens_lists[seq_id].begin(),
                                        tokens_lists[seq_id].begin() + n_prefill);

        if (decode(ctx, ctx_sampling, model,
                   batch, batch_tokens, batch_embd, params.n_batch, n_embd,
                   seq_id, tokens,
                   0, n_prefill,
                   hidden_states, 0,
                   std::vector<llama_token>{},
                   out_tokens) != 0) {
            LOG_TEE("%s: prefill() failed\n", __func__);
            return 1;
        }

        // dump the kv-cache
        export_kv_cache_buffers(
            ctx_params, ctx, model,
            0, n_prefill,
            0, n_layers - params.n_skip_layers,
            seq_id, prefix_caches);

        // not used any more
        llama_kv_cache_seq_rm(ctx, seq_id, 0, n_prefill);

        int layer_to_export = n_layers - params.n_skip_layers;
        uint64_t current_buffer_size = n_prefill
            * layer_to_export
            * static_cast<uint64_t>(k_tensor_size + v_tensor_size);

        uint64_t offset = 0;
        for (llama_pos p = 0; p < n_prefill; ++p) {
            uint32_t token_id = tokens[p];
            // printf("%s: seq_id: %u, token_id = %u\n", __func__, i, token_id);
            out.write(reinterpret_cast<char *>(&token_id), 4);
            out.write(reinterpret_cast<char *>(&layer_to_export), 4);
            for (int32_t l = 0; l < layer_to_export; ++l) {
                // write key
                out.write(reinterpret_cast<char *>(&k_tensor_size), 4);
                out.write(reinterpret_cast<char *>(prefix_caches.data()) + offset, k_tensor_size);
                offset += k_tensor_size;
                // write value
                out.write(reinterpret_cast<char *>(&v_tensor_size), 4);
                out.write(reinterpret_cast<char *>(prefix_caches.data()) + offset, v_tensor_size);
                offset += v_tensor_size;
            }
        }
        if (offset != current_buffer_size) {
            LOG_TEE("%s: inconsistent: offset = %zu, current_buffer_size = %zu\n",
                    __func__, offset, current_buffer_size);
            return 1;
        }
        LOG_TEE("%s: exported seqeunce %u: n_tokens = %u, n_layers = %u, current_buffer_size = %zu\n",
                __func__, seq_id, n_prefill, n_layers, current_buffer_size);
    }

    out.flush();
    out.close();

    llama_print_timings(ctx);

    fprintf(stderr, "\n");

    llama_batch_free(batch);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
