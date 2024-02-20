# llama.cpp/example/suri

The purpose of this example is to demonstrate a minimal usage of suri.cpp for generating text with a given prompt (with collaboration with the vLLM server).

```bash
./bin/suri ~/models/llama-2-70b.AWQ.fp16.Q4_0.gguf "Hello my name is" 0 32

...

llama_build_graph: non-view tensors processed: 1684/1684
llama_new_context_with_model: compute buffer total size = 311.19 MiB
export_kv_cache_buffers: allocate kv_cache buffer size = 19005440
export_kv_cache_buffers: export kv cache: n_ctx = 2048, n_embd_gpa = 1024, kv head = 58, kv size = 2048, kv used = 58
export_kv_cache_buffers: type_k = f16, type_v = f16
Connected
Received: 10
Received: {"code":0}
Received: 21
Received: {"code":0,"text":"."}

...
```
