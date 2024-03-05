#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "llm-cache/ds/kv_state_cache_manager.h"

static int import_kv_cache_buffers(
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
    if (sequence_length == 0) {
        LOG_TEE("%s: skip non-cached sequence: %u\n", __func__, seq_id);
        return 0;
    }

    if (layer0 >= layer1) {
        LOG_TEE("%s: error: invalid layer range: [%d, %d)\n", __func__, layer0, layer1);
        return -1;
    }

    uint32_t k_cache_elements = llama_model_n_embd_k_gqa(model);
    uint32_t v_cache_elements = llama_model_n_embd_v_gqa(model);
    uint32_t k_tensor_nbytes = ggml_row_size(ctx_params.type_k, k_cache_elements);
    uint32_t v_tensor_nbytes = ggml_row_size(ctx_params.type_v, v_cache_elements);
    uint32_t n_layers = llama_model_n_layer(model);
    uint64_t buffer_nbytes = sequence_length
            * n_layers
            * static_cast<uint64_t>(k_tensor_nbytes + v_tensor_nbytes);

    if (import_kv_cache_buffers(ctx, buffer.data(), buffer_nbytes,
                                seq_id, token_start_pos, token_stop_pos, 0, n_layers,
                                repermute_k) != 0) {
        LOG_TEE("%s: import_kv_cache_buffers() failed\n", __func__);
        return -1;
    }
    LOG_TEE("%s: imported: sequence_length = %u (%u -> %u), n_layers = %d (%d -> %d), current_buffer_size = %zu\n",
            __func__, sequence_length, token_start_pos, token_stop_pos,
            layer1 - layer0, layer0, layer1, buffer_nbytes);
    return 0;
}

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

static inline std::string escape(const std::string &str) {
    std::string escaped;
    for (char c : str) {
        if (c == '\n') {
            escaped += "\\n";
        } else if (c == '\r') {
            escaped += "\\r";
        } else if (c == '\t') {
            escaped += "\\t";
        } else {
            escaped += c;
        }
    }
    return escaped;
}

int main(int argc, char ** argv) {
    gpt_params params;
    int with_vineyard = 0;

    if (argc < 4 || argv[1][0] == '-') {
        printf("usage: %s MODEL_PATH [PROMPT OR PROMPT_FILE] [N_GPU_LAYERS] [N_PREDICT] [N_BATCH]\n" , argv[0]);
        return 1 ;
    }

    if (argc >= 2) {
        params.model = argv[1];
    }

    if (argc >= 3) {
        params.prompt = argv[2];
    }

    if (argc >= 4) {
        with_vineyard = std::atoi(argv[3]);
    }

    if (argc >= 5) {
        params.n_gpu_layers = std::atoi(argv[4]);
    }

    if (argc >= 6) {
        params.n_predict = std::atoi(argv[5]);
    }

    if (argc >= 7) {
        params.n_batch = std::atoi(argv[6]);
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

    ctx_params.offload_kqv = false;
    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }
    uint32_t n_ctx = llama_n_ctx(ctx);
    uint32_t n_embd = llama_model_n_embd(model);
    uint32_t n_layers = llama_model_n_layer(model);

    // sampling context
    params.sparams.temp = 1.0f;  // 0.0 for greedy and 1.0 for random
    struct llama_sampling_context * ctx_sampling = llama_sampling_init(params.sparams);

    // tokenize the prompt
    std::vector<std::vector<llama_token>> tokens_lists(prompts.size());
    int max_token_list_size = tokenize(ctx, prompts, tokens_lists);
    llama_seq_id tokens_lists_size = tokens_lists.size();

    LOG_TEE("%s: max_token_list_size = %d\n", __func__, max_token_list_size);
    // create a llama_batch with size 512
    // we use this object to submit token data for decoding
    llama_batch batch = llama_batch_init(params.n_batch, 0, prompts.size());
    llama_token *batch_tokens = batch.token;
    float *batch_embd = (float *) malloc(sizeof(float) * params.n_batch * n_embd);

    std::vector<uint8_t> prefix_caches;

    std::vector<std::vector<uint8_t>> hidden_states;
    std::vector<llama_token> out_tokens;
    std::string output_content;

    uint32_t k_cache_elements = llama_model_n_embd_k_gqa(model);
    uint32_t v_cache_elements = llama_model_n_embd_k_gqa(model);
    uint32_t k_tensor_nbytes = ggml_row_size(ctx_params.type_k, k_cache_elements);
    uint32_t v_tensor_nbytes = ggml_row_size(ctx_params.type_v, v_cache_elements);

    uint64_t max_buffer_size = max_token_list_size
            * n_layers
            * static_cast<uint64_t>(k_tensor_nbytes + v_tensor_nbytes);
    // TODO: reserve a continuous buffer, will be fixed after we extend the `import/export` to support non-continuous buffers
    prefix_caches.resize(max_buffer_size);

    if (k_tensor_nbytes != v_tensor_nbytes) {
        LOG_TEE("%s: error: inconsistent: k_tensor_nbytes = %u, v_tensor_nbytes = %u\n",
                __func__, k_tensor_nbytes, v_tensor_nbytes);
        return 1;
    }

    // init vineyard kv cache
    const int tensor_bytes = k_tensor_nbytes;
    const int capacity = 2048;
    const int block_size = 128;

    std::shared_ptr<KVStateCacheManager> manager;
    if (with_vineyard != 0){
        manager = std::make_shared<KVStateCacheManager>(tensor_bytes, capacity, n_layers, block_size);
    }
    std::vector<std::map<int, std::pair<LLMKV, LLMKV>>> kv_state_list;

    for (llama_seq_id seq_id = 0; seq_id < tokens_lists_size; seq_id++) {
        kv_state_list.clear();
        llama_pos n_prefill = 0;
        std::vector<llama_token> prefix_tokens;
        uint64_t offset = 0;
        if (with_vineyard != 0){
            manager->Query(tokens_lists[seq_id], kv_state_list);
            n_prefill = kv_state_list.size();

            // copy the kv-cache to the buffer, as `import_kv_cache_buffers` requires
            // a continuous buffer currently.
            for (llama_pos prefill_pos = 0; prefill_pos < n_prefill; ++prefill_pos){
                for (uint32_t il = 0; il < n_layers; il++) {
                    LLMKV key_state = kv_state_list[prefill_pos][il].first;
                    LLMKV value_state = kv_state_list[prefill_pos][il].second;
                    memcpy(prefix_caches.data() + offset, key_state.data, k_tensor_nbytes);
                    offset += k_tensor_nbytes;
                    memcpy(prefix_caches.data() + offset, value_state.data, v_tensor_nbytes);
                    offset += v_tensor_nbytes;
                }
            }

            uint64_t buffer_nbytes = n_prefill * n_layers * static_cast<uint64_t>(k_tensor_nbytes + v_tensor_nbytes);
            if (offset != buffer_nbytes) {
                LOG_TEE("%s: inconsistent: offset = %zu, buffer_nbytes = %zu\n",
                        __func__, offset, buffer_nbytes);
                return 1;
            }
            LOG_TEE("%s: matched %u prefix tokens from vineyard\n", __func__, n_prefill);

            // import kv-cache
            if (n_prefill > 0) {
                llama_batch_clear(batch);

                batch.token = batch_tokens;
                batch.embd = nullptr;

                for (llama_pos i = 0; i < n_prefill; i++) {
                    llama_batch_add(batch,
                                    tokens_lists[seq_id][i],
                                    i,
                                    { seq_id },
                                    false);
                }
                if (llama_allocate_kvcache_slots(ctx, batch) != 0) {
                    LOG_TEE("%s: llama_allocate_kvcache_slots() failed\n", __func__);
                    return 1;
                }
            }
            if (import_kv_cache_buffers(
                ctx_params, ctx, model,
                0, n_prefill,
                0, n_layers - params.n_skip_layers,
                seq_id, prefix_caches) != 0) {
                LOG_TEE("%s: import_kv_cache_buffers() failed\n", __func__);
                return 1;
            }
        }
        // reset the context
        out_tokens.clear();
        output_content.clear();

        std::vector<llama_token> tokens(tokens_lists[seq_id].begin() + n_prefill,
                                        tokens_lists[seq_id].end());
        llama_pos start_pos = n_prefill;
        llama_pos stop_pos = tokens_lists[seq_id].size();
        std::vector<llama_token> decode_outputs;

        // auto-regressive decoding
        while ((params.n_predict == -1 && (out_tokens.empty() || out_tokens.back() != llama_token_eos(model))) || (params.n_predict > static_cast<int32_t>(out_tokens.size()))) {
            if (decode(ctx, ctx_sampling, model,
                       batch, batch_tokens, batch_embd, params.n_batch, n_embd,
                       seq_id, tokens,
                       start_pos, stop_pos,
                       hidden_states, 0,
                       std::vector<llama_token>{tokens.back()},
                       decode_outputs) != 0) {
                LOG_TEE("%s: prefill/decode() failed\n", __func__);
                return 1;
            }
            if (decode_outputs.empty()) {
                LOG_TEE("%s: error: decode_outputs is empty\n", __func__);
                return 1;
            }

            // print the decoded tokens
            llama_token token_id = decode_outputs.back();
            out_tokens.push_back(token_id);
            output_content += escape(llama_token_to_piece(ctx, token_id));
            fprintf(stderr, "Output[%2d][%3zu][%5u]: %s\n", seq_id, out_tokens.size(), token_id, output_content.c_str());

            // prepare for the next decoding round
            tokens.clear();
            tokens.emplace_back(token_id);
            start_pos = stop_pos;
            stop_pos = start_pos + 1;
        }
        if (with_vineyard != 0){
            // export kv-cache
            llama_pos token_start_pos = n_prefill;
            llama_pos token_stop_pos = tokens_lists[seq_id].size();
            export_kv_cache_buffers(
                ctx_params, ctx, model,
                token_start_pos, token_stop_pos,
                0, n_layers,
                seq_id, prefix_caches);

            LOG_TEE("%s: updating kv cache in vineyard: %u -> %u\n", __func__, token_start_pos, token_stop_pos);
            // updates the remaining kv-cache into vineyard
            offset = 0;
            for (llama_pos i = token_start_pos; i < token_stop_pos; ++i) {
                std::map<int, std::pair<LLMKV, LLMKV>> new_kv_state;
                // compose `new_kv_state`
                for (uint32_t il = 0; il < n_layers; il++) {
                    new_kv_state[il].first.data = prefix_caches.data() + offset;
                    offset += k_tensor_nbytes;
                    new_kv_state[il].first.length = k_tensor_nbytes;
                    new_kv_state[il].second.data = prefix_caches.data() + offset;
                    offset += v_tensor_nbytes;
                    new_kv_state[il].second.length = v_tensor_nbytes;
                }
                manager->Update(prefix_tokens, tokens_lists[seq_id][i], new_kv_state);
                // move on
                prefix_tokens.push_back(tokens_lists[seq_id][i]);
            }
            LOG_TEE("%s: updated kv cache in vineyard: %u -> %u\n", __func__, token_start_pos, token_stop_pos);
        }
        // not used any more
        llama_kv_cache_seq_rm(ctx, seq_id, 0, n_prefill);
    }

    llama_print_timings(ctx);

    fprintf(stderr, "\n");

    llama_batch_free(batch);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
