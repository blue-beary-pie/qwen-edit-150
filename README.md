# ComfyUI 图像编辑 API (多 GPU 并行版)

## 项目概述

本项目是一个基于 FastAPI 构建的高性能图像编辑 API 服务。它集成了 ComfyUI 核心功能，通过多进程架构实现在多个 GPU 上的并行推理。该服务专门针对大规模并发请求进行了优化，支持动态任务分发、自动显存管理及故障恢复。

## 核心特性

*   **多 GPU 高并发**：支持在多个 CUDA 设备上运行多个独立 worker 进程。
*   **灵活配置**：通过 `.env` 文件轻松配置每个 GPU 的并发 worker 数量。
*   **智能任务分发**：主进程自动将任务推送到任务队列，由空闲的 worker 竞争处理。
*   **异常恢复机制**：
    *   监控 worker 进程状态，异常退出或显存溢出 (OOM) 时自动重启。
    *   失败的任务会自动重新排队处理。
*   **推理优化**：
    *   **启动延迟**：每个 worker 启动间隔 15 秒，避免瞬间 IO 激增卡死服务器。
    *   **显存优化**：使用 `torch.no_grad()` 和 `expandable_segments` 降低内存开销。
    *   **模型热加载**：模型加载后常驻显存，响应速度快。
*   **强大的 URL 兼容性**：自动处理包含空格或特殊字符的图像 URL。

## 目录结构

```text
/mnt/data0/AIGC/qwen-edit-248/
├── api_service.py      # FastAPI 主服务，负责 API 路由、任务分发与进程监控
├── worker.py           # Worker 进程逻辑，执行 ComfyUI 推理
├── .env                # 配置文件（GPU 并发数等）
├── requirements.txt    # 项目依赖
├── README.md           # 本文档
```

## 安装与配置

### 1. 确认 ComfyUI 路径
服务默认使用以下路径的 ComfyUI：
`/mnt/data0/AIGC/story/comfyui_main`

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置 GPU 并发
编辑 `.env` 文件：
```ini
# 格式：GPU_ID:并发数,GPU_ID:并发数
# 示例：GPU 0 跑 1 个，GPU 1 跑 2 个，GPU 2 跑 2 个
GPU_CONCURRENCY_CONFIG=0:1,1:2,2:2
```

## 运行服务

```bash
# 默认端口 18002
python api_service.py
```

## API 使用说明

### 图像编辑接口

*   **URL**: `POST /image-edit`
*   **Content-Type**: `application/json`

#### 请求参数 (JSON)

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `image1` | `string` | 否 | `null` | 输入图像的 URL 或本地绝对路径 |
| `prompt` | `string` | 是 | - | 图像生成或编辑的提示词 |
| `steps` | `int` | 否 | `4` | 扩散步数 |
| `width` | `int` | 否 | `null` | 输出图像宽度（不传则自动识别） |
| `height` | `int` | 否 | `null` | 输出图像高度（不传则自动识别） |
| `seed` | `int` | 否 | 随机 | 随机种子 |
| `cfg` | `float` | 否 | `1.0` | 引导比例 |
| `sampler_name`| `string` | 否 | `euler_ancestral`| 采样器名称 |
| `scheduler` | `string` | 否 | `beta` | 调度器名称 |

#### 示例请求

```json
{
  "image1": "http://example.com/input.png",
  "prompt": "老人，头上开花",
  "steps": 4,
  "width": 1024,
  "height": 1024
}
```

#### 示例响应

```json
{
  "status": "success",
  "image_url": "http://<ip>:18002/outputs/abc-123.png"
}
```

## 开发备注

*   **隔离性**：每个 worker 通过 `CUDA_VISIBLE_DEVICES` 实现设备隔离。
*   **安全性**：输出图像 Tensor 已执行 `.detach()` 处理，确保转换 Numpy 时不会触发梯度错误。
*   **监控**：控制台会实时输出 worker 的状态和推理耗时。
