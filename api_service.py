import sys
import os
import json
import uuid
import requests
import io
import asyncio
import time
import multiprocessing
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import random
import uvicorn
import logging
from PIL import Image
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 从 worker.py 导入 worker 进程函数
from worker import run_worker as worker_process

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ComfyUI-API")

# --- 主应用程序配置 ---

# ComfyUI 的安装路径
COMFY_PATH = "/mnt/data0/AIGC/story/comfyui_main"
# ComfyUI 输出图像的目录
OUTPUT_DIR = os.path.join(COMFY_PATH, "output")
# 如果输出目录不存在，则创建它
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 初始化 FastAPI 应用程序，设置标题
app = FastAPI(title="ComfyUI Image Edit API (Multi-GPU)")
# 将 ComfyUI 的输出目录挂载为静态文件服务，可以通过 /outputs/{filename} 访问
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# 定义图像编辑请求的数据模型
class ImageEditRequest(BaseModel):
    image1: Optional[str] = None  # 输入图像的 URL 或本地路径，可选
    prompt: str  # 图像编辑的提示词
    steps: Optional[int] = 4  # 扩散步数，可选，默认为 4
    width: Optional[int] = None  # 输出图像宽度，可选
    height: Optional[int] = None  # 输出图像高度，可选
    seed: Optional[int] = None  # 随机种子，可选，如果为 None 则自动生成
    cfg: float = 1.0  # 分类器自由引导比例，默认为 1.0
    sampler_name: Optional[str] = "euler_ancestral"  # 采样器名称，可选，默认为 "euler_ancestral"
    scheduler: Optional[str] = "beta"  # 调度器名称，可选，默认为 "beta"

# --- GPU 配置从 .env 获取 ---
# 默认配置为 0:1,1:2,2:2
gpu_config_str = os.getenv("GPU_CONCURRENCY_CONFIG", "0:1,1:2,2:2")
GPUS = []
try:
    # 解析 "0:1,1:2,2:2" 格式为 [0, 1, 1, 2, 2]
    for part in gpu_config_str.split(","):
        if ":" in part:
            gpu_id, count = part.split(":")
            GPUS.extend([int(gpu_id)] * int(count))
except Exception as e:
    logger.error(f"Failed to parse GPU_CONCURRENCY_CONFIG '{gpu_config_str}': {e}. Falling back to default.")
    GPUS = [0, 1, 1, 2, 2]

logger.info(f"Final GPU configuration: {GPUS}")

# 用于进程管理的全局变量
input_queue = None  # 输入队列，用于向 worker 进程发送任务
output_queue = None  # 输出队列，用于从 worker 进程接收结果
workers = {}  # 修改为字典: {worker_idx: Process}
response_futures = {}  # 存储每个请求的 asyncio.Future 对象，用于异步等待结果

# 存储待处理任务的数据，用于 OOM 重试
pending_tasks = {} 

@app.on_event("startup")
async def startup_event():
    """
    FastAPI 应用程序启动时触发的事件。
    初始化多进程队列和 worker 进程。
    """
    global input_queue, output_queue
    
    # 设置多进程启动方法为 'spawn'，以确保与 PyTorch 兼容性
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # 如果已经设置过，则忽略错误
        
    # 创建用于进程间通信的队列
    input_queue = multiprocessing.Queue()  # 任务输入队列
    output_queue = multiprocessing.Queue() # 结果输出队列
    
    logger.info(f"Starting workers for configuration: {GPUS}")
    # 为每个配置的 GPU 启动对应的 worker 进程
    for i, gpu_id in enumerate(GPUS):
        start_worker(i, gpu_id)
        # 增加启动延迟，避免 IO 瞬间激增导致服务器卡死
        if i < len(GPUS) - 1:
            logger.info(f"Waiting 15 seconds before starting next worker to reduce IO burst...")
            await asyncio.sleep(15)
    
    # 启动结果收集器作为后台任务
    asyncio.create_task(result_collector())
    # 启动进程监控器
    asyncio.create_task(worker_monitor())

def start_worker(worker_idx, gpu_id):
    """启动或重启指定的 worker 进程"""
    global workers
    if worker_idx in workers and workers[worker_idx].is_alive():
        logger.info(f"Worker {worker_idx} on GPU {gpu_id} is already running.")
        return

    logger.info(f"Starting/Restarting worker {worker_idx} on GPU {gpu_id}...")
    p = multiprocessing.Process(
        target=worker_process,
        args=(gpu_id, input_queue, output_queue, COMFY_PATH),
        name=f"Worker-{worker_idx}-GPU-{gpu_id}"
    )
    p.start()
    workers[worker_idx] = p

async def worker_monitor():
    """后台监控 worker 进程状态，异常退出时自动重启"""
    while True:
        try:
            for i, gpu_id in enumerate(GPUS):
                if i not in workers or not workers[i].is_alive():
                    logger.warning(f"Worker {i} on GPU {gpu_id} is dead. Restarting...")
                    start_worker(i, gpu_id)
                    # 重启后增加延迟，避免多个 worker 同时重启导致 IO 激增
                    logger.info(f"Waiting 15 seconds after restart...")
                    await asyncio.sleep(15)
        except Exception as e:
            logger.error(f"Worker monitor error: {e}")
        await asyncio.sleep(5) # 每 5 秒检查一次状态

@app.on_event("shutdown")
async def shutdown_event():
    """
    FastAPI 应用程序关闭时触发的事件。
    向所有 worker 进程发送终止信号并等待它们结束。
    """
    logger.info("Shutting down workers...")
    # 向每个 worker 进程发送 None 信号，表示终止
    for _ in range(len(GPUS)):
        input_queue.put(None)
    # 等待所有 worker 进程完成
    for worker_idx, p in workers.items():
        p.join()

async def result_collector():
    """
    后台任务，用于从 output_queue 收集结果并解析对应的 Future 对象。
    支持 OOM 任务重试。
    """
    logger.info("Result collector started.")
    while True:
        try:
            loop = asyncio.get_running_loop()
            # 将阻塞的 queue.get() 操作放到线程池中执行，避免阻塞事件循环
            result = await loop.run_in_executor(None, output_queue.get)
            
            request_id, response_data = result
            
            # 处理 OOM 情况
            if response_data.get('status') == 'oom':
                logger.warning(f"Request {request_id} failed due to OOM. Re-queueing...")
                if request_id in pending_tasks:
                    # 将任务重新放入队列，它会被其他空闲的 worker 拾取
                    input_queue.put((request_id, pending_tasks[request_id]))
                continue # 不要解析 Future，让它继续等待重试结果

            # 如果请求 ID 存在于 response_futures 中，则设置 Future 的结果
            if request_id in response_futures:
                # 任务完成后，从待处理任务中删除
                pending_tasks.pop(request_id, None)
                future = response_futures.pop(request_id)
                if not future.done(): # 确保 Future 尚未完成
                    future.set_result(response_data)
            else:
                logger.warning(f"Received result for unknown request {request_id}")
                
        except Exception as e:
            logger.error(f"Result collector error: {e}")
            await asyncio.sleep(0.1)

def download_image(url):
    """
    根据 URL 或本地路径下载或读取图像数据。
    支持 HTTP/HTTPS URL 和本地文件路径。
    """
    try:
        # 预处理 URL：去除可能存在的首尾空格和引号
        if isinstance(url, str):
            url = url.strip().strip('`').strip('"').strip("'")
            
        if url.startswith("http"):
            # 如果是 HTTP/HTTPS URL，则使用 requests 下载
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, timeout=30, headers=headers)
            response.raise_for_status() # 检查 HTTP 错误
            img_data = response.content
        elif os.path.exists(url):
            # 如果是本地文件路径，则直接读取文件
            with open(url, "rb") as f:
                img_data = f.read()
        else:
            raise ValueError(f"Invalid URL or path: {url}")
        
        # 验证下载或读取的数据是否为有效图像
        try:
            Image.open(io.BytesIO(img_data)).convert("RGB")
        except Exception as e:
            # 如果 URL 指向 HTML 页面而不是图像，则抛出特定错误
            if url.startswith("http") and 'text/html' in response.headers.get('Content-Type', ''):
                raise ValueError("URL points to an HTML page, not a direct image.")
            raise e # 重新抛出其他图像验证错误
            
        return img_data
    except Exception as e:
        logger.error(f"Error loading image from {url}: {e}")
        # 抛出 HTTPException，以便 FastAPI 能够捕获并返回给客户端
        raise HTTPException(status_code=400, detail=f"Failed to load image: {e}")

@app.post("/image-edit")
async def image_edit(request_data: ImageEditRequest, request: Request):
    """
    处理图像编辑请求。
    1. 下载输入图像。
    2. 将图像编辑任务提交给 worker 进程。
    3. 等待 worker 进程返回结果。
    4. 保存生成的图像并返回其 URL。
    """
    # 1. 在主进程中下载图像 (I/O 密集型操作)
    # 为了避免阻塞 GIL，将下载操作放到线程池中执行
    loop = asyncio.get_running_loop()
    img_data = None
    if request_data.image1:
        try:
            img_data = await loop.run_in_executor(None, download_image, request_data.image1)
        except HTTPException as e:
            raise e # 重新抛出 download_image 中抛出的 HTTPException
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) # 捕获其他异常并返回 500 错误

    # 2. 提交任务给 worker 进程
    request_id = str(uuid.uuid4()) # 生成唯一的请求 ID
    future = loop.create_future() # 创建一个 Future 对象用于等待结果
    response_futures[request_id] = future # 将 Future 存储起来，以便 result_collector 可以设置其结果
    
    # 如果没有提供 seed，则生成一个随机种子
    current_seed = request_data.seed if request_data.seed is not None else random.randint(0, 2**32 - 1)
    
    # 准备传递给 worker 进程的任务参数
    task_params = {
        'image_data': img_data,
        'prompt': request_data.prompt,
        'steps': request_data.steps,
        'width': request_data.width,
        'height': request_data.height,
        'seed': current_seed,
        'cfg': request_data.cfg,
        'sampler_name': request_data.sampler_name,
        'scheduler': request_data.scheduler
    }
    
    # 将任务存入 pending_tasks 以便 OOM 时重试
    pending_tasks[request_id] = task_params
    
    input_queue.put((request_id, task_params)) # 将任务放入输入队列

    # 3. 等待结果
    try:
        # 设置超时，等待 worker 进程处理完成
        result = await asyncio.wait_for(future, timeout=300.0) # 300 秒超时
    except asyncio.TimeoutError:
        response_futures.pop(request_id, None) # 超时时从 futures 中移除
        raise HTTPException(status_code=504, detail="Processing timed out") # 返回 504 超时错误
        
    if result['status'] == 'error':
        raise HTTPException(status_code=500, detail=result['message']) # worker 进程返回错误

    # 4. 保存并响应
    result_bytes = result['data'] # 获取 worker 进程返回的图像字节数据
    filename = f"{uuid.uuid4()}.png" # 生成唯一的 PNG 文件名
    filepath = os.path.join(OUTPUT_DIR, filename) # 构造文件保存路径
    
    # 将结果字节写入文件
    with open(filepath, "wb") as f:
        f.write(result_bytes)
        
    # 构造生成图像的完整 URL
    base_url = str(request.base_url).rstrip('/')
    image_url = f"{base_url}/outputs/{filename}"
    
    # 返回包含图像 URL 和文件名的 JSON 响应
    return JSONResponse(content={
        "image_url": image_url,
        "filename": filename
    })

if __name__ == "__main__":
    # 当直接运行此脚本时，启动 Uvicorn 服务器
    uvicorn.run(app, host="0.0.0.0", port=18002)
