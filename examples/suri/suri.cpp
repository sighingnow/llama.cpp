#include "common.h"
#include "llama.h"

#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "hv/WebSocketClient.h"
#include "hv/json.hpp"

using json = nlohmann::json;

enum class StatusCode {
    ERROR = -1,
    INITIATED = 0,
    SUBMITTED = 1,
    CONTINUED = 2,
    FINISHED = 3,
};

template <typename T>
class block_queue_t {
public:
    block_queue_t() : size_limit_(std::numeric_limits<size_t>::max()) {}
    ~block_queue_t() {}

    void push(const T& item) {
        {
            std::unique_lock<std::mutex> lk(lock_);
            while (queue_.size() >= size_limit_) {
                full_.wait(lk);
            }
            queue_.emplace_back(item);
        }
        empty_.notify_one();
    }

    void push(T&& item) {
        {
            std::unique_lock<std::mutex> lk(lock_);
            while (queue_.size() >= size_limit_) {
                full_.wait(lk);
            }
            queue_.emplace_back(std::move(item));
        }
        empty_.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lk(lock_);
        while (queue_.empty()) {
            empty_.wait(lk);
        }
        T rc(std::move(queue_.front()));
        queue_.pop_front();
        full_.notify_one();
        return rc;
    }

    void get(T& item) {
        std::unique_lock<std::mutex> lk(lock_);
        while (queue_.empty()) {
            empty_.wait(lk);
        }
        item = std::move(queue_.front());
        queue_.pop_front();
        full_.notify_one();
    }

    size_t size() const { return queue_.size(); }

    void clear() { queue_.clear(); }
private:
    std::deque<T> queue_;
    size_t size_limit_;
    std::mutex lock_;
    std::condition_variable empty_, full_;
};

class web_socket_client_t {
public:
    web_socket_client_t(std::string const &endpoint) {
        client_lock = std::unique_lock<std::mutex>(client_mutex);
        client.onopen = [&]() {
            std::cout << "Connected" << std::endl;
        };
        client.onclose = [&]() {
            std::cout << "Disconnected" << std::endl;
            disconnected.store(true);
            client_cv.notify_all();

            // add an empty message to ensure outer loop can exit
            messages.push("");
        };
        client.onmessage = [&](const std::string &message) {
            messages.push(message);
            client_cv.notify_all();
        };

        disconnected.store(false);
        client.open(endpoint.c_str());
    }

    void send(const std::string &str) {
        client.send(str);
    }

    void send_buffer(const void *buffer,  size_t buffer_nbytes, size_t chunk_nbytes = 65535/* the fragment value from libhv */) {
        json meta;
        meta["buffer_nbytes"] = buffer_nbytes;
        meta["chunk_nbytes"] = chunk_nbytes;
        client.send(meta.dump());
        for (size_t i = 0; i < buffer_nbytes; i += chunk_nbytes) {
            size_t len = std::min(chunk_nbytes, buffer_nbytes - i);
            client.send(reinterpret_cast<const char *>(buffer) + i, len);
        }
    }

    std::string recv() {
        return messages.pop();
    }

    size_t recv_buffer(std::vector<uint8_t> &buffer) {
        std::string message = messages.pop();
        json meta = json::parse(message);
        size_t buffer_nbytes = meta["buffer_nbytes"];
        size_t chunk_nbytes = meta["chunk_nbytes"];

        buffer.resize(buffer_nbytes);
        for (size_t i = 0; i < buffer_nbytes; i += chunk_nbytes) {
            std::string message = messages.pop();
            buffer.resize(message.size());
            std::copy(message.begin(), message.end(), buffer.begin() + i);
        }
        return buffer_nbytes;
    }

    void close() {
        client.close();
    }

    bool closed() {
        return disconnected.load();
    }

    void wait() {
        client_cv.wait_for(client_lock, std::chrono::seconds(1), [&]() {
            return disconnected.load();
        });
    }

private:
    std::mutex client_mutex;
    std::unique_lock<std::mutex> client_lock;
    std::condition_variable client_cv;

    hv::WebSocketClient client;
    std::atomic_bool disconnected;

    block_queue_t<std::string> messages;
};

static std::tuple<size_t /* buffer_nbytes */,
                  uint32_t /* num_layers */,
                  uint32_t /* tensor_nbytes */> export_kv_cache_buffers(
    llama_context_params ctx_params,
    llama_context * ctx,
    llama_model *model,
    llama_pos token_start_pos,
    llama_pos token_stop_pos,
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

    uint32_t k_cache_elements = llama_model_n_embd_k_gqa(model);
    uint32_t v_cache_elements = llama_model_n_embd_v_gqa(model);
    uint32_t k_tensor_nbytes = ggml_row_size(ctx_params.type_k, k_cache_elements);
    uint32_t v_tensor_nbytes = ggml_row_size(ctx_params.type_v, v_cache_elements);
    uint32_t n_layers = llama_model_n_layer(model);
    uint64_t buffer_nbytes = sequence_length
            * n_layers
            * static_cast<uint64_t>(k_tensor_nbytes + v_tensor_nbytes);
    LOG_TEE("%s: allocate kv_cache buffer size = %zu\n", __func__, buffer_nbytes);
    buffer.resize(buffer_nbytes);

    if (export_kv_cache_buffers(ctx, buffer.data(), buffer_nbytes,
                                seq_id, token_start_pos, token_stop_pos, 0, n_layers,
                                repermute_k) != 0) {
        LOG_TEE("%s: export_kv_cache_buffers() failed\n", __func__);
        return std::make_tuple(0, 0, 0);
    }
    LOG_TEE("%s: exported: sequence_length = %u (%u -> %u), n_layers = %u, current_buffer_size = %zu\n",
            __func__, sequence_length, token_start_pos, token_stop_pos, n_layers, buffer_nbytes);
    return std::make_tuple(buffer_nbytes, n_layers, k_tensor_nbytes);
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
    std::string endpoint = "localhost:8000";

    llama_pos n_prompt_chunk = 16;
    uint32_t collaborative = 16;

    if (argc == 1 || argv[1][0] == '-') {
        printf("usage: %s MODEL_PATH [PROMPT OR PROMPT_FILE] [N_PROMPT_CHUNK] [N_GPU_LAYERS] [N_SKIP_LAYERS] [N_PREDICT] [ENDPOINT]\n" , argv[0]);
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
        params.n_predict = std::atoi(argv[6]);
    }

    if (argc >= 8) {
        endpoint = argv[7];
    }
    endpoint = "ws://" + endpoint + "/ws/generate";

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
    llama_model *model = llama_load_model_from_file(params.model.c_str(), model_params);

    if (model == NULL) {
        LOG_TEE( "%s: error: unable to load model\n" , __func__);
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
    uint32_t n_ctx = llama_n_ctx(ctx);
    uint32_t n_embd = llama_model_n_embd(model);

    // sampling context
    params.sparams.temp = 0.0f;  // greedy
    struct llama_sampling_context * ctx_sampling = llama_sampling_init(params.sparams);

    if (ctx == NULL) {
        LOG_TEE( "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // tokenize the prompt
    std::vector<std::vector<llama_token>> tokens_lists(prompts.size());
    int max_token_list_size = tokenize(ctx, prompts, tokens_lists);

    // create a llama_batch with size 512
    // we use this object to submit token data for decoding
    llama_batch batch = llama_batch_init(params.n_batch, 0, prompts.size());
    llama_token *batch_tokens = batch.token;
    float *batch_embd = (float *) malloc(sizeof(float) * params.n_batch * n_embd);

    // select the first sequence
    llama_seq_id seq_id = 0;
    std::vector<std::vector<uint8_t>> hidden_states;

    std::vector<llama_token> out_tokens;
    std::string output;

    std::string prompt = prompts[seq_id];
    llama_pos n_tokens = tokens_lists[seq_id].size();
    // the last two chunk won't be prefilled by the client
    llama_pos n_prefill = n_prompt_chunk * std::max(
        0, (n_tokens + n_prompt_chunk - 1) / n_prompt_chunk - 2);
    llama_pos n_prefill_progress = 0;
    std::vector<uint8_t> prefix_caches;

    // connect to server
    web_socket_client_t client(endpoint);

    // first step
    json response;
    response["code"] = static_cast<int>(StatusCode::INITIATED);

    // main loop
    do {
        StatusCode code = static_cast<StatusCode>(
            response.value("code", static_cast<int>(StatusCode::ERROR)));

        if (code == StatusCode::ERROR) {
            LOG_TEE("error: unexpected code %d\n", static_cast<int>(code));
            break;
        }

        if (code == StatusCode::INITIATED && n_prefill_progress < n_prefill) {
            std::vector<llama_token> tokens(tokens_lists[seq_id].begin() + n_prefill_progress,
                                            tokens_lists[seq_id].begin() + n_prefill_progress + n_prompt_chunk);

            if (decode(ctx, ctx_sampling, model,
                    batch, batch_tokens, batch_embd, params.n_batch, n_embd,
                    seq_id, tokens,
                    n_prefill_progress, n_prefill_progress + n_prompt_chunk,
                    hidden_states, 0,
                    std::vector<llama_token>{},
                    out_tokens) != 0) {
                LOG_TEE("%s: prefill() failed\n", __func__);
                return 1;
            }

            // dump the kv-cache
            auto tup = export_kv_cache_buffers(
                ctx_params, ctx, model,
                n_prefill_progress, n_prefill_progress + n_prompt_chunk,
                seq_id, prefix_caches);
            size_t buffer_nbytes = std::get<0>(tup);
            uint32_t num_layers = std::get<1>(tup);
            uint32_t tensor_nbytes = std::get<2>(tup);

            json request;
            request["type"] = "initiate";
            request["buffer_nbytes"] = buffer_nbytes;
            request["num_layers"] = num_layers;
            request["tensor_nbytes"] = tensor_nbytes;
            request["prefix_tokens"] = tokens;
            client.send(request.dump());

            // send kv-cache buffer
            client.send_buffer(prefix_caches.data(), buffer_nbytes);

            // update prefill progress
            n_prefill_progress += n_prompt_chunk;
        } else if (code == StatusCode::INITIATED && n_prefill_progress >= n_prefill) {
            // skip some layers before collaborating
            llama_model_skip_layers(model, params.n_skip_layers);

            json request;
            request["type"] = "submit";
            request["prompt"] = prompt;
            request["temperature"] = 0.0;
            if (params.n_predict != -1) {
                request["max_tokens"] = params.n_predict;
            } else {
                request["max_tokens"] = n_ctx - tokens_lists[seq_id].size();
            }
            request["sequence_length"] = n_tokens;
            request["collaborative"] = collaborative;
            client.send(request.dump());
        } else if (code == StatusCode::SUBMITTED || code == StatusCode::CONTINUED) {
            std::string text = response["text"];
            std::vector<llama_token> hidden_state_tokens = response["hidden_state_tokens"];
            llama_pos hidden_state_position = response["hidden_state_position"];
            std::vector<llama_token> candidate_tokens = response["candidate_tokens"];
            size_t hidden_state_nbytes = response["hidden_state_nbytes"];

            // receive hidden state tensors
            if (hidden_states.size() < hidden_state_tokens.size()) {
                hidden_states.resize(hidden_state_tokens.size());
            }
            for (size_t i = 0; i < hidden_state_tokens.size(); i++) {
                hidden_states[i].resize(hidden_state_nbytes);
                client.recv_buffer(hidden_states[i]);
            }

            // no need for further decoding
            if (candidate_tokens.size() == 0) {
                break;
            }

            if (decode(ctx, ctx_sampling, model,
                       batch, batch_tokens, batch_embd, params.n_batch, n_embd,
                       seq_id, hidden_state_tokens,
                       hidden_state_position, hidden_state_position + hidden_state_tokens.size(),
                       hidden_states, hidden_state_tokens.size(),
                       candidate_tokens,
                       out_tokens) != 0) {
                LOG_TEE("%s: decode() failed\n", __func__);
                break;
            }
            for (auto const &token : out_tokens) {
                output += escape(llama_token_to_piece(ctx, token));
            }
            LOG_TEE("Output[%3zu][%zu]: %s\n", output.size(), out_tokens.size(), output.c_str());

            // continue generation
            json request;
            request["type"] = "continue";
            request["rollback"] = candidate_tokens.size();
            request["revised_output_tokens"] = out_tokens;
            client.send(request.dump());
        } else if (code == StatusCode::FINISHED) {
            break;
        } else {
            // unexpected status code
            printf("Received unknown status code: %d\n", static_cast<int>(code));
            break;
        }

        // read the next message
        std::string message = client.recv();
        // std::cout << "message: '" << message << "'" << std::endl;
        if (message.empty()) {
            break;
        }
        response = json::parse(message);
    } while (!client.closed());

    client.close();
    client.wait();

    llama_print_timings(ctx);

    fprintf(stderr, "\n");

    llama_batch_free(batch);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
