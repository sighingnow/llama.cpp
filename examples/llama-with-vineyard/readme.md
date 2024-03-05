
1. Open a terminal to start vineyard server

```shell
./bin/vineyardd --socket /tmp/vineyard_test.sock
```

2. Open a terminal to start llama-with-vineyard
```shell
export VINEYARD_IPC_SOCKET=/tmp/vineyard_test.sock
./bin/llama-with-vineyard /opt/tao/models/llama-2-7b-chat.Q4_0.gguf ../examples/llama-with-vineyard/prompts.txt 1 0 64
```