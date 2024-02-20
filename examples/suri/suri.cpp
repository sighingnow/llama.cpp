#include "common.h"
#include "llama.h"

#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "hv/WebSocketClient.h"
#include "hv/json.hpp"

using json = nlohmann::json;

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
            std::cout << "Received: " << message.size() << std::endl;
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
                  uint32_t /* sequence_length */,
                  uint32_t /* num_layers */,
                  uint32_t /* tensor_nbytes */> export_kv_cache_buffers(
    llama_context_params ctx_params,
    llama_context * ctx,
    llama_model *model,
    const std::vector<llama_token> &tokens_list,
    llama_seq_id seq_id,
    std::vector<uint8_t> &buffer)
{
    // buffer schema: [[(k tensor, v tensor) * layers] * sequence_length]
    //
    // see also:
    // - https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py#L133-L135
    // - https://github.com/ggerganov/llama.cpp/blob/master/convert.py#L565
    size_t repermute_k = llama_model_n_embd_head(model);
    uint32_t sequence_length = static_cast<uint32_t>(tokens_list.size());

    uint32_t n_cache_elements = llama_model_n_embd_gqa(model);
    uint32_t tensor_nbytes = ggml_row_size(ctx_params.type_v, n_cache_elements);
    uint32_t n_layers = llama_model_n_layer(model);
    uint64_t buffer_nbytes = static_cast<uint64_t>(tensor_nbytes)
        * n_layers
        * sequence_length
        * 2 /* k & v */;
    printf("%s: allocate kv_cache buffer size = %zu\n", __func__, buffer_nbytes);
    buffer.resize(buffer_nbytes);

    export_kv_cache_buffers(ctx, buffer.data(), seq_id, 0, sequence_length, 0, n_layers, repermute_k);
    return std::make_tuple(buffer_nbytes, sequence_length, n_layers, tensor_nbytes);
}

static int prefill(
    llama_context * ctx,
    const std::vector<std::string> &prompts,
    std::vector<std::vector<llama_token>> &tokens_lists
) {
    // tokenize the prompt
    const bool add_bos = true;  // default, unconfigurable setting in vLLM.

    size_t num_tokens = 0;
    size_t max_token_list_size = 0;
    tokens_lists.resize(prompts.size());
    for (size_t i = 0; i < prompts.size(); i++) {
        tokens_lists[i] = ::llama_tokenize(ctx, prompts[i], add_bos);
        num_tokens += tokens_lists[i].size();
        max_token_list_size = std::max(max_token_list_size, tokens_lists[i].size());
    }
    llama_seq_id tokens_lists_size = static_cast<int>(tokens_lists.size());

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
    return llama_decode(ctx, batch);
}

int main(int argc, char ** argv) {
    gpt_params params;
    std::string endpoint = "localhost:8000";

    if (argc == 1 || argv[1][0] == '-') {
        printf("usage: %s MODEL_PATH [PROMPT OR PROMPT_FILE] [N_GPU_LAYERS] [N_PREDICT] [ENDPOINT]\n" , argv[0]);
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
        params.n_predict = std::atoi(argv[4]);
    }

    if (argc >= 6) {
        endpoint = argv[5];
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
    llama_backend_init(params.numa);

    // initialize the model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;
    llama_model *model = llama_load_model_from_file(params.model.c_str(), model_params);

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

    // create a llama_batch with size 512
    // we use this object to submit token data for decoding
    llama_batch batch = llama_batch_init(params.n_batch, 0, prompts.size());

    // prefill
    std::vector<std::vector<llama_token>> tokens_lists(prompts.size());
    if (prefill(ctx, prompts, tokens_lists) != 0) {
        fprintf(stderr, "%s: prefill() failed\n", __func__);
        return 1;
    }

    // select the first sequence
    llama_seq_id seq_id = 0;

    // dump the kv-cache
    std::vector<llama_token> prefix_cached = tokens_lists[seq_id];
    std::vector<uint8_t> prefix_caches;
    auto tup = export_kv_cache_buffers(ctx_params, ctx, model, prefix_cached, seq_id, prefix_caches);
    size_t buffer_nbytes = std::get<0>(tup);
    uint32_t sequence_length = std::get<1>(tup);
    uint32_t num_layers = std::get<2>(tup);
    uint32_t tensor_nbytes = std::get<3>(tup);

    // connect to server
    web_socket_client_t client(endpoint);

    // initiate
    {
        json request;
        request["type"] = "initiate";
        request["prompt"] = prompts[0];
        request["temperature"] = 0.0;
        request["max_tokens"] = params.n_predict;
        request["buffer_nbytes"] = buffer_nbytes;
        request["sequence_length"] = sequence_length;
        request["num_layers"] = num_layers;
        request["tensor_nbytes"] = tensor_nbytes;
        request["prefix_cached"] = prefix_cached;
        client.send(request.dump());

        // send kv-cache buffer
        client.send_buffer(prefix_caches.data(), prefix_caches.size());
    }

    // generate
    while (!client.closed()) {
        std::string message = client.recv();
        std::cout << "Received: " << message << std::endl;
        if (message.empty()) {
            break;
        }
        // continue generation
        json request;
        request["type"] = "continue";
        client.send(request.dump());
    }
    client.wait();

    llama_print_timings(ctx);

    fprintf(stderr, "\n");

    llama_batch_free(batch);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
