# ComfyUI 图像编辑 API (多 GPU)

## 项目概述

这是一个基于 FastAPI 构建的图像编辑 API 服务，它利用 ComfyUI 的强大功能在多个 GPU 上并行处理图像编辑任务。该服务旨在提供一个高效、可扩展的接口，用于通过 API 请求执行图像生成和编辑操作。

## 特性

*   **多 GPU 支持**：通过多进程架构，将图像编辑任务分发到多个 GPU 上并行执行，提高处理吞吐量。
*   **自动分发**：任务会自动分发到空闲的 CUDA 设备上处理。
*   **异常处理与自动重启**：监控 worker 进程状态。如果某个 GPU 发生显存溢出 (OOM) 或异常退出，系统会自动重启该 GPU 的服务，并将失败的任务重新分发给其他可用的 GPU。
*   **异步处理**：利用 FastAPI 的异步能力和 `asyncio`，实现非阻塞的请求处理和结果收集。
*   **任务队列**：使用 `multiprocessing.Queue` 实现主进程与 worker 进程之间的任务分发和结果回收。
*   **ComfyUI 集成**：利用 ComfyUI 进行实际的图像生成和编辑操作。
*   **静态文件服务**：生成的图像可以通过 API 提供的 URL 直接访问。
*   **灵活的图像输入**：支持通过 URL 或本地文件路径提供输入图像。

## 目录结构

```text
comfyui_api/
├── api_service.py      # FastAPI 服务入口，负责任务分发和监控
├── worker.py           # Worker 进程逻辑，负责在特定 GPU 上执行 ComfyUI 推理
├── ComfyUI/            # ComfyUI 核心库 (已在 .gitignore 中忽略，需自行部署)
│   ├── main.py         # ComfyUI 入口
│   ├── models/         # 模型存放目录
│   └── output/         # 生成图像存放目录
├── .gitignore          # 忽略 ComfyUI 及缓存文件
├── README.md           # 项目说明文档
└── 需求prompt历史.txt    # 需求变更历史记录
```

## 设置

### 1. 克隆 ComfyUI

确保您已将 ComfyUI 克隆到项目根目录下的 `ComfyUI` 文件夹中。

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git ComfyUI
```

### 2. 安装依赖

```bash
pip install -r requirements.txt # 假设存在一个 requirements.txt 文件，包含 FastAPI, uvicorn, requests, Pillow 等
pip install fastapi uvicorn requests Pillow
```

### 3. 配置 ComfyUI

确保您的 ComfyUI 环境已正确配置，并且所有必要的模型和自定义节点已安装。

### 4. GPU 配置

您可以通过环境变量 `USE_GPUS` 来指定要使用的 GPU ID，支持多种格式。默认使用 `1,2,3,4,5,6,7`。

```bash
# 使用 1, 2, 3 号卡
export USE_GPUS="1,2,3"
# 或者使用带 cu 前缀的格式
export USE_GPUS="cu3,cu4,cu5"
# 启动服务
python api_service.py
```

## 运行服务

```bash
python api_service.py
```

服务将在 `http://0.0.0.0:18002` 上启动。

## API 端点

### `POST /image-edit`

处理图像编辑请求。

**请求体 (JSON)**:

```json
{
    "image1": "string",         // 可选：输入图像的 URL 或本地文件路径。如果不提供，将进行纯文本生成。
    "prompt": "string",         // 必填：图像生成或编辑的提示词
    "steps": 4,                 // 可选：扩散步数 (默认: 4)
    "width": null,              // 可选：输出图像宽度 (默认: null, ComfyUI 自动处理)
    "height": null,             // 可选：输出图像高度 (默认: null, ComfyUI 自动处理)
    "seed": null,               // 可选：随机种子 (默认: null, 自动生成随机数)
    "cfg": 1.0,                 // 可选：分类器自由引导比例 (默认: 1.0)
    "sampler_name": "euler_ancestral", // 可选：采样器名称 (默认: "euler_ancestral")
    "scheduler": "beta"         // 可选：调度器名称 (默认: "beta")
}
```

**响应 (JSON)**:

```json
{
    "image_url": "string",      // 生成图像的完整 URL
    "filename": "string"        // 生成图像的文件名
}
```

**错误响应**:

*   `400 Bad Request`: 图像加载失败或请求参数无效。
*   `500 Internal Server Error`: 服务器内部错误或 worker 进程处理失败。
*   `504 Gateway Timeout`: 图像处理超时。

## 使用示例

```python
import requests
import json

api_url = "http://localhost:18002/image-edit"

payload = {
    "image1": "https://example.com/your_image.jpg", # 替换为您的图像 URL 或本地路径
    "prompt": "a cat with a hat, highly detailed, fantasy art",
    "steps": 4,
    "seed": 12345,
    "cfg": 7.0,
    "sampler_name": "dpmpp_2m",
    "scheduler": "karras"
}

headers = {
    "Content-Type": "application/json"
}

try:
    response = requests.post(api_url, data=json.dumps(payload), headers=headers)
    response.raise_for_status() # 检查 HTTP 错误
    result = response.json()
    print("Generated Image URL:", result["image_url"])
    print("Filename:", result["filename"])
except requests.exceptions.RequestException as e:
    print(f"API request failed: {e}")
    if response is not None:
        print("Response content:", response.text)

```