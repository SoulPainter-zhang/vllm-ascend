# # 快速入门

## 环境准备

本文档以 Atlas 800I A2 推理服务器和 Qwen3.6‑27B 模型为例，让开发者快速开始使用 VLLM 进行大模型推理流程。

### 前提条件

物理机部署场景，需要在物理机安装 NPU 驱动固件以及部署 Docker，执行如下步骤判断是否已安装 NPU 驱动固件和部署 Docker。

- 执行以下命令查看 NPU 驱动固件是否安装。若出现类似如图 1 所示，说明已安装。否则请参见表 1 进行安装。

```
npu‑smi info
```

图 1 回显信息

表 1 Atlas A2 推理系列产品

表格

| 产品型号 | 参考文档 |
| --- | --- |
| Atlas 800I A2 | 《Atlas A2 中心推理和训练硬件 24.1.0 NPU 驱动和固件安装指南》中的 “物理机安装与卸载” 章节 |

- 执行以下命令查看 Docker 是否已安装并启动。

```
docker ps
```

回显以下信息表示 Docker 已安装并启动。

```
CONTAINER ID        IMAGE        COMMAND         CREATED        STATUS         PORTS           NAMES
```

### 获取模型权重

1. 请先下载权重，这里以 Qwen3.6‑27B 为例，下载链接：千问 3.6‑27B・模型库，将权重文件上传至服务器任意目录（如 /home/weight）。
2. 执行以下命令，修改权重文件权限：

```
chmod -R 755 /home/weight
```

### 获取容器镜像

进入昇腾官方镜像仓库，根据设备型号选择下载对应的 VLLM 镜像。

该镜像已具备模型运行所需的基础环境，包括：CANN、FrameworkPTAdapter、VLLM 与 VLLM‑Ascend，可实现模型快速上手推理。

表 2 容器内各组件安装路径

表格

| 组件 | 安装路径 |
| --- | --- |
| CANN | /usr/local/Ascend/cann |
| CANN‑NNAL‑ATB | /usr/local/Ascend/nnal/atb |
| VLLM | /vllm‑workspace/vllm |
| VLLM‑Ascend | /vllm‑workspace/vllm‑ascend |

## 启动容器

1. 下载完成镜像后，执行以下命令启动容器。

```
docker run -itd --privileged --name=<container-name> --ipc=host --net=host  --shm-size 500g  \
--device=/dev/davinci0  --device=/dev/davinci1  --device=/dev/davinci2  --device=/dev/davinci3  \
--device=/dev/davinci4  --device=/dev/davinci5  --device=/dev/davinci6  --device=/dev/davinci7  \
--device=/dev/davinci_manager  --device=/dev/hisi_hdc  --device /dev/devmm_svm  \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
-v /usr/local/sbin/npu‑smi:/usr/local/sbin/npu‑smi  -v /usr/local/sbin:/usr/local/sbin \
-v /etc/hccn.conf:/etc/hccn.conf -v /home/weight:/home/weight  fe1e88748912
```

> 
> [!NOTE] 说明
> 
> 
> - “fe1e88748912” 为镜像 ID，可根据实际情况修改。

表 1 参数说明

表格

| 参数 | 参数说明 |
| --- | --- |
| --name | 设置容器名称。 |
| --device | 表示映射的设备，可以挂载一个或者多个设备。需要挂载的设备如下：/dev/davinciX：NPU 设备，X 是 ID 号，如：davinci0。/dev/davinci_manager：davinci 相关的管理设备。/dev/hisi_hdc：hdc 相关管理设备。/dev/devmm_svm：内存管理相关设备。可根据以下命令查询 device 个数及名称方式，根据需要绑定设备，修改上面命令中的 "--device=****"。`ll /dev/` |
| -v /usr/local/Ascend/driver:/usr/local/Ascend/driver | 将宿主机目录 “/usr/local/Ascend/driver” 挂载到容器，请根据驱动所在实际路径修改。 |
| -v /path‑to‑weights:/path‑to‑weights | 设定权重挂载的路径，需要根据用户的情况修改。请将权重文件和数据集文件同时放置于该路径下。 |

2. 执行以下命令进入容器。

```
docker exec -it <container-name> bash
```

## 模型推理

1. 进入容器后，执行以下命令启动服务。

```
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export LD_PRELOAD=/usr/lib/aarch64‑linux‑gnu/libjemalloc.so.2:$LD_PRELOAD
export TASK_QUEUE_ENABLE=1
export CODEBASE_DIR="/vllm‑workspace"
export PYTHONPATH="${CODEBASE_DIR}/vllm‑ascend:${CODEBASE_DIR}/vllm:${PYTHONPATH}"
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export VLLM_ASCEND_ENABLE_NZ=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_ASCEND_GDN_FAST_PATH=1
export VLLM_ASCEND_GDN_MAX_PADDING_RATIO=2.0
export VLLM_ASCEND_GDN_MAX_H_OVERALLOC_RATIO=2.0

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

vllm serve /home/weight/Qwen3.6‑27B \
    --served‑model‑name "qwen3.6‑27b" \
    --host 0.0.0.0 \
    --port 8314 \
    --tensor‑parallel‑size 8 \
    --max‑model‑len 262144 \
    --max‑num‑batched‑tokens 8192 \
    --max‑num‑seqs 196 \
    --gpu‑memory‑utilization 0.95 \
    --trust‑remote‑code \
    --async‑scheduling \
    --allowed‑local‑media‑path / \
    --mm_processor_cache_type="shm" \
    --mm‑processor‑cache‑gb 0 \
    --speculative‑config '{"num_speculative_tokens": 3, "method":"qwen3_5_mtp", "enforce_eager": true}' \
    --additional‑config '{"enable_cpu_binding":true, "multistream_overlap_shared_expert": true, "enable_weight_nz_layout":true}' \
    --compilation‑config '{"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[4,8,12,16,20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284, 288, 292, 296, 300, 304, 308, 312, 316, 320, 324, 328, 332, 336, 340, 344, 348, 352, 356, 360, 364, 368, 372, 376, 380, 384, 388, 392, 396, 400, 404, 408, 412, 416, 420, 424, 428, 432, 436, 440, 444, 448, 452, 456, 460, 464, 468, 472, 476, 480, 484, 488, 492, 496, 500, 504, 508, 512, 516, 520, 524, 528, 532, 536, 540, 544, 548, 552, 556, 560, 564, 568, 572, 576, 580, 584, 588, 592, 596, 600, 604, 608, 612, 616, 620, 624, 628, 632, 636, 640, 644, 648, 652, 656, 660, 664, 668, 672, 676, 680, 684, 688, 692, 696, 700, 704, 708, 712, 716, 720, 724, 728, 732, 736, 740, 744, 748, 752, 756, 760, 764, 768, 772, 776, 780, 784, 788, 792, 796, 800, 804, 808, 812, 816, 820, 824, 828, 832, 836, 840, 844, 848, 852, 856, 860, 864, 868, 872, 876, 880, 884, 888, 892, 896, 900, 904, 908, 912, 916, 920, 924, 928, 932, 936, 940, 944, 948, 952, 956, 960, 964, 968, 972, 976, 980, 984, 988, 992, 996, 1000]}'
```

根据实际情况修改启动命令中的配置参数，参数说明如下

表格

| 配置项 | 配置说明 |
| --- | --- |
| ASCEND_RT_VISIBLE_DEVICES | 表示启用哪几张卡。对于每个模型实例分配的 npuIds，使用芯片逻辑 ID 表示。 |
| vllm serve | 模型权重路径。程序会读取该路径下的 config.json 中 torch_dtype 和 vocab_size 字段的值，需保证路径和相关字段存在。必填，默认值："/data/atb_testdata/weights/llama1‑65b‑safetensors"。该路径会进行安全校验，需要和执行用户的属组和权限保持一致。 |
| served‑model‑name | 模型名称。 |
| host | 模型服务 IP，0.0.0.0 表示不指定 IP 地址，该服务器的 IP 皆可访问。 |
| port | 模型服务端口号。 |
| max‑model‑len | 模型服务上下文长度。 |

2. 发送请求。
用户可使用 HTTPS 客户端（Linux curl 命令，Postman 工具等）发送 HTTPS 请求，此处以 Linux curl 命令为例进行说明。
重开一个窗口，使用以下命令发送请求。例如验证服务是否拉起：

```
curl http://<IP>:<Port>/v1/chat/completions \
  -H "Content‑Type: application/json" \
  -d '{
    "model": "qwen3.6‑27b",
    "messages": [
      {
        "role": "user",
        "content": "who are you"
      }
    ],
    "max_tokens": 2048,
    "temperature": 0
  }' | python3 -m json.tool
```

回显如下则表明请求发送成功：

```
{
    "id": "chatcmpl‑984264b9b96eee5d",
    "object": "chat.completion",
    "created": 1787584707,
    "model": "qwen3.6‑27B",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Here's a thinking process:\n\n1.  **Analyze User Input:** The user asks \"who are you\"\n2.  **Identify Core Intent:** The user wants to know my identity, capabilities, and origin.\n3.  **Check Internal Guidelines/Constraints:**\n   - I am Qwen (通义千问), developed by Alibaba Group's Tongyi Lab.\n   - I should respond clearly, concisely, and truthfully about my identity.\n   - I should maintain a helpful and professional tone.\n4.  **Formulate Response:**\n   - Acknowledge the question directly.\n   - State my identity: I'm Qwen, a large language model developed by Alibaba Group's Tongyi Lab.\n   - Briefly mention my purpose/capabilities: I'm designed to assist with a wide range of tasks like answering questions, writing, coding, analysis, and more.\n   - Keep it concise and open to further questions.\n5.  **Draft Response (Mental Refinement):**\n   I'm Qwen, a large language model developed by Alibaba Group's Tongyi Lab. I'm designed to help with a wide variety of tasks—whether that's answering questions, writing, coding, analyzing data, or just having a thoughtful conversation. How can I assist you today?\n6.  **Final Check against Guidelines:** Matches identity, concise, helpful tone, no extra fluff. Ready.✅\n\n\nI'm Qwen, a large language model independently developed by Alibaba Group's Tongyi Lab. I'm designed to assist with a wide range of tasks, including answering questions, writing, coding, logical reasoning, data analysis, and more. How can I help you today?",
                "refusal": null,
                "annotations": null,
                "audio": null,
                "function_call": null,
                "reasoning": null
            },
            "logprobs": null,
            "finish_reason": "stop",
            "stop_reason": null,
            "token_ids": null,
            "routed_experts": null
        }
    ],
    "service_tier": null,
    "system_fingerprint": "vllm‑0.23.0‑pp2‑4559800a",
    "usage": {
        "prompt_tokens": 13,
        "total_tokens": 372,
        "completion_tokens": 359,
        "prompt_tokens_details": null,
        "completion_tokens_details": null
    },
    "prompt_logprobs": null,
    "prompt_token_ids": null,
    "prompt_text": null,
    "kv_transfer_params": null
}
```

## 精度测试 (不需要)

> 
> [!NOTE] 说明
> 
> 
> - 精度测试和性能测试前，请先重开一个窗口进入容器，并参见 3 设置环境变量。
> - 以下精度测试以 AISBench 工具为例，AISBench 工具的详细使用方法请参见 AISBench 工具。

1. 使用以下命令下载并安装 AISBench 工具。

```
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark/
pip3 install -e ./ --use‑pep517
pip3 install -r requirements/api.txt
pip3 install -r requirements/extra.txt
```

> 
> [!NOTE] 说明
> pip 安装方式适用于使用 AISBench 最新功能的场景（镜像安装 MindIE 方式除外）。AISBench 工具已预装在 MindIE 镜像中，可使用以下命令查看 AISBench 工具在 MindIE 镜像中的安装路径。
> 
> 
> ```
> pip show ais_bench_benchmark
> ```

2. 准备数据集。

以 gsm8k 为例，单击 gsm8k 数据集下载数据集，将解压后的 gsm8k 文件夹部署到工具根路径的 ais_bench/datasets 文件夹下。

3. 配置 ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py 文件，示例如下所示。

```
from ais_bench.benchmark.models import VLLMCustomAPIChatStream
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm‑api‑stream‑chat',
        path="/home/weight/Qwen3.6‑27B",                    # 指定模型序列化词表文件绝对路径，一般来说就是模型权重文件夹路径
        model="qwen3.6‑27b",        # 指定服务端已加载模型名称，依据实际VLLM推理服务拉取的模型名称配置（配置成空字符串会自动获取）
        request_rate = 0,           # 请求发送频率，每1/request_rate秒发送1个请求给服务端，小于0.1则一次性发送所有请求
        retry = 2,
        host_ip = "127.0.0.1",      # 指定推理服务的IP
        host_port = 8314,           # 指定推理服务的端口
        max_out_len = 512,          # 推理服务输出的token的最大数量
        batch_size=1,               # 请求发送的最大并发数
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature = 0.5,
            top_k = 10,
            top_p = 0.95,
            seed = None,
            repetition_penalty = 1.03,
        ) ,
         pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
```

4. 执行以下命令启动服务化精度测试。

```
ais_bench --models vllm_api_stream_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --debug
```

回显如下所示则表示执行成功：

```
dataset                 version  metric   mode  vllm_api_general_chat
----------------------- -------- -------- ----- ----------------------
demo_gsm8k              401e4c   accuracy gen                   62.50
```

## 性能测试（不需要）

> 
> [!NOTE] 说明
> 以下性能测试以 AISBench 工具为例，AISBench 工具的详细使用方法请参见 AISBench 工具。

1. 使用以下命令下载并安装 AISBench 工具。

```
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark/
pip3 install -e ./ --use‑pep517
pip3 install -r requirements/api.txt
pip3 install -r requirements/extra.txt
```

> 
> [!NOTE] 说明
> pip 安装方式适用于使用 AISBench 最新功能的场景（镜像安装 MindIE 方式除外）。AISBench 工具已预装在 MindIE 镜像中，可使用以下命令查看 AISBench 工具在 MindIE 镜像中的安装路径。
> 
> 
> ```
> pip show ais_bench_benchmark
> ```

2. 准备数据集。

以 gsm8k 为例，单击 gsm8k 数据集下载数据集，将解压后的 gsm8k / 文件夹部署到工具根路径的 ais_bench/datasets 文件夹下。

3. 配置 ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py 文件，示例如下所示。

```
from ais_bench.benchmark.models import VLLMCustomAPIChatStream
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm‑api‑stream‑chat',
        path="/home/weight/Qwen3.6‑27B",                    # 指定模型序列化词表文件绝对路径，一般来说就是模型权重文件夹路径
        model="qwen3.6‑27b",        # 指定服务端已加载模型名称，依据实际VLLM推理服务拉取的模型名称配置（配置成空字符串会自动获取）
        request_rate = 0,           # 请求发送频率，每1/request_rate秒发送1个请求给服务端，小于0.1则一次性发送所有请求
        retry = 2,
        host_ip = "127.0.0.1",      # 指定推理服务的IP
        host_port = 1025,           # 指定推理服务的端口
        max_out_len = 512,          # 推理服务输出的token的最大数量
        batch_size=1,               # 请求发送的最大并发数
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature = 0.5,
            top_k = 10,
            top_p = 0.95,
            seed = None,
            repetition_penalty = 1.03,
            ignore_eos = True,      # 推理服务输出忽略eos（输出长度一定会达到max_out_len）
        ) ,
         pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
```

4. 执行以下命令启动服务化性能测试。

```
ais_bench --models vllm_api_stream_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --mode perf --debug
```

回显如下所示则表示执行成功：

```
╒════════════╤════╤════════╤═══════╤══════╤═══════╤══════╤═══════╤═══════╤════╤═══════╤═══╤════╤═══════╤═══╤════╤═══════╤═══╤════╤═══════╤═══╤
│ Performance Parameters │ Stage  │ Average        │ Min          │ Max        │ Median       │ P75        │ P90          │ P99          │ N │
│ E2EL                   │total   │ 2048.2945  ms  │ 1729.7498 ms │ 3450.96 ms │ 2491.8789 ms │ 2750.85 ms │ 3184.9186 ms │ 3424.4354 ms │ 8 │
│ TTFT                   │total   │ 50.332 ms      │ 50.6244 ms   │ 52.0585 ms │ 50.3237 ms   │ 50.5872 ms │ 50.7566 ms   │ 50 .0551 ms  │ 8 │
│ TPOT                   │total   │ 10.6965 ms     │ 10.061 ms    │ 10.8805 ms │ 10.7495 ms   │ 10.7818 ms │ 10.808 ms    │ 10.8582 ms   │ 8 │
│ ITL                    │total   │ 10.6965 ms     │ 7.3583 ms    │ 13.7707 ms │ 10.7513 ms   │ 10.8009 ms │ 10.8358 ms   │ 10.9322 ms   │ 8 │
│ InputTokens            │total   │ 1512.5         │ 1481.0       │ 1566.0     │ 1511.5       │ 1520.25    │ 1536.6       │ 1563.06      │ 8 │
│ OutputTokens           │total   │ 287.375        │ 200.0        │ 407.0      │ 280.0        │ 322.75     │ 374.8        │ 403.78       │ 8 │
│ OutputTokenThroughput  │total   │ 115.9216       │ 107.6555     │ 116.5352   │ 117.6448     │ 118.2426   │ 118.3765     │ 118.6388     │ 8 │
╘════════════╧════╧════════╧═══════╧══════╧═══════╧══════╧═══════╧═══════╧═══╧═══════╧═══════╧═══╧═══════╧═══════╧═══╧═══════╧═══════╧═══╧═══╧

╒═════════════╤═════╤══════════╤═════╤══════════╤═════╤════╕
│ Common Metric            │ Stage    │ Value              │
│ Benchmark Duration       │ total    │ 19897.8505 ms      │
│ Total Requests           │ total    │ 8                  │
│ Failed Requests          │ total    │ 0                  │
│ Success Requests         │ total    │ 8                  │
│ Concurrency              │ total    │ 0.9972             │
│ Max Concurrency          │ total    │ 1                  │
│ Request Throughput       │ total    │ 0.4021 req/s       │
│ Total Input Tokens       │ total    │ 12100              │
│ Prefill Token Throughput │ total    │ 17014.3123 token/s │
│ Total generated tokens   │ total    │ 2299               │
│ Input Token Throughput   │ total    │ 608.7438 token/s   │
│ Output Token Throughput  │ total    │ 115.7835 token/s   │
│ Total Token Throughput   │ total    │ 723.5273 token/s   │
╘═════════════╧═════╧══════════╧══════════╧══════════╧═════╛
```

性能测试结果主要关注 TTFT、TPOT、Request Throughput 和 Output Token Throughput 输出参数。

> 
> [!NOTE] 说明
> 任务执行的过程最终会落盘在默认的输出路径，该输出路径在运行中的打印日志中有提示，日志内容如下所示：
> 
> 
> ```
> 08/28 15:13:26 - AISBench - INFO - Current exp folder: outputs/default/20250828_151326
> ```

命令执行结束后，outputs/default/20250828_151326 中的任务执行的详情如下所示：

```
20250828_151326           # 每次实验基于时间戳生成的唯一目录
├── configs               # 自动存储的所有已转储配置文件
├── logs                  # 执行过程中日志，命令中如果加--debug，不会有过程日志落盘（都直接打印出来了）
│   └── performance/      # 推理阶段的日志文件
└── performance           # 性能测评结果
    └── vllm‑api‑stream‑chat/         # “服务化模型配置”名称，对应模型任务配置文件中models的 abbr参数
        ├── gsm8kdataset.csv          # 单次请求性能输出（CSV），与性能结果打印中的Performance Parameters表格一致
        ├── gsm8kdataset.json         # 端到端性能输出（JSON），与性能结果打印中的Common Metric表格一致
        ├── gsm8kdataset_details.json # 全量打点日志（JSON）
        └── gsm8kdataset_plot.html    # 请求并发可视化报告（HTML）
```