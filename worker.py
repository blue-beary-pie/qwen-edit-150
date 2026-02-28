import os
import sys
import time
import io
import numpy as np
from PIL import Image

def run_worker(gpu_id, input_queue, output_queue, comfy_path):
    """
    Worker process that runs ComfyUI inference on a specific GPU.
    Crucially, this function must set CUDA_VISIBLE_DEVICES before importing torch.
    """
    try:
        # Set the CUDA device for this process BEFORE importing torch
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Optimization: Reduce fragmentation and enable better memory management
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        print(f"Worker {os.getpid()}: Set CUDA_VISIBLE_DEVICES={gpu_id}")

        # Add ComfyUI to path
        if comfy_path not in sys.path:
            sys.path.insert(0, comfy_path)

        # Now import torch and ComfyUI modules
        import torch
        
        # Verify isolation
        visible_devices = torch.cuda.device_count()
        print(f"Worker {os.getpid()}: Torch sees {visible_devices} device(s).")
        
        if visible_devices > 1:
            print(f"Worker {os.getpid()}: WARNING - Isolation failed! Setting device manually.")
            try:
                torch.cuda.set_device(gpu_id)
            except Exception as e:
                print(f"Worker {os.getpid()}: ERROR setting device: {e}")
        else:
            # If 1 device is visible, it is mapped to cuda:0
            print(f"Worker {os.getpid()}: Isolation successful. Using cuda:0 (physical GPU {gpu_id})")

        import folder_paths
        import nodes
        import comfy.model_management
        import comfy.utils
        from comfy_extras.nodes_qwen import TextEncodeQwenImageEditPlus
        
        # --- Optimization: Force high VRAM usage and disable aggressive unloading ---
        try:
            # Set to NORMAL_VRAM first to be safe, then try HIGH_VRAM if we have room
            # NORMAL_VRAM will still try to keep models in VRAM but is more careful
            comfy.model_management.vram_state = comfy.model_management.VRAMState.NORMAL_VRAM
            print(f"Worker {os.getpid()}: VRAM State set to NORMAL_VRAM")
            
            # Clear cache to ensure we start clean
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Worker {os.getpid()}: Warning - Could not set VRAM state: {e}")
        
        # Configuration
        CKPT_NAME = "Qwen-Rapid-AIO-SFW-v23.safetensors"
        MODELS_DIR = os.path.join(comfy_path, "models/checkpoints")
        
        folder_paths.add_model_folder_path("checkpoints", MODELS_DIR)
        
        print(f"Worker {os.getpid()}: Loading model {CKPT_NAME}...")
        
        # Load model
        loader = nodes.CheckpointLoaderSimple()
        model, clip, vae = loader.load_checkpoint(CKPT_NAME)
        
        # --- Optimization: Pre-warm models to GPU ---
        # This forces the weights into VRAM now, so the first request is fast
        try:
            print(f"Worker {os.getpid()}: Pre-warming models to GPU...")
            # If CLIP object doesn't support the attribute, try loading only the main model
            try:
                comfy.model_management.load_models_gpu([model, clip, vae])
            except:
                comfy.model_management.load_models_gpu([model])
            
            # After loading, try to switch to HIGH_VRAM if possible for max performance
            # but only if we didn't OOM during loading
            comfy.model_management.vram_state = comfy.model_management.VRAMState.HIGH_VRAM
            print(f"Worker {os.getpid()}: Successfully switched to HIGH_VRAM")
        except Exception as e:
            print(f"Worker {os.getpid()}: Warning - Pre-warming failed or could not set HIGH_VRAM: {e}")
        
        # Initialize nodes
        sampler_node = nodes.KSampler()
        vae_decode_node = nodes.VAEDecode()
        empty_latent_node = nodes.EmptyLatentImage()
        
        # --- Optimization: Warm up with a dummy inference ---
        # This ensures all JIT kernels and dynamic loaders are ready
        print(f"Worker {os.getpid()}: Performing dummy warm-up inference...")
        try:
            torch.cuda.empty_cache()
            with torch.no_grad():
                # Just a very small, fast inference (32x32 to minimize activation VRAM)
                dummy_latent = empty_latent_node.generate(width=32, height=32, batch_size=1)[0]
                
                # Perform a 1-step sample to force ALL models into VRAM (CLIP, VAE, Model)
                # We need simple conditioning for this
                dummy_pos = TextEncodeQwenImageEditPlus.execute(clip=clip, prompt="warm up", vae=vae).result
                dummy_neg = TextEncodeQwenImageEditPlus.execute(clip=clip, prompt="", vae=vae).result
                
                # Format conditioning
                if isinstance(dummy_pos, tuple): dummy_pos = dummy_pos[0]
                if isinstance(dummy_neg, tuple): dummy_neg = dummy_neg[0]
                if not isinstance(dummy_pos, list): dummy_pos = [dummy_pos]
                if not isinstance(dummy_neg, list): dummy_neg = [dummy_neg]

                # 1 step sampler to force load everything
                sampler_node.sample(
                    model=model,
                    seed=1,
                    steps=1,
                    cfg=1.0,
                    sampler_name="euler",
                    scheduler="simple",
                    positive=dummy_pos,
                    negative=dummy_neg,
                    latent_image=dummy_latent,
                    denoise=1.0
                )
                print(f"Worker {os.getpid()}: Warm-up inference completed.")
        except Exception as e:
            print(f"Worker {os.getpid()}: Warning - Warm-up inference failed: {e}")
        
        print(f"Worker {os.getpid()}: Ready to process requests.")
        
        while True:
            try:
                # Get request from queue
                req_data = input_queue.get()
                if req_data is None: # Sentinel to stop
                    break
                
                request_id, params = req_data
                print(f"Worker {os.getpid()}: Processing request {request_id}")
                
                start_time = time.time()
                
                # Unpack params
                image_data = params['image_data'] # Bytes
                prompt = params['prompt']
                steps = params['steps']
                width = params['width']
                height = params['height']
                seed = params['seed']
                cfg = params['cfg']
                sampler_name = params['sampler_name']
                scheduler = params['scheduler']
                
                # Process Image
                img_comfy = None
                if image_data:
                    img = Image.open(io.BytesIO(image_data)).convert("RGB")
                    orig_width, orig_height = img.size
                    
                    # Determine target dimensions
                    target_width = width or orig_width
                    target_height = height or orig_height
                    
                    # Convert to tensor
                    img_np = np.array(img).astype(np.float32) / 255.0
                    img_comfy = torch.from_numpy(img_np)[None,]
                else:
                    # Default dimensions if no image provided
                    target_width = width or 1024
                    target_height = height or 1024
                
                # Ensure dimensions are multiples of 8
                target_width = (target_width // 8) * 8
                target_height = (target_height // 8) * 8
                
                # Execution
                # Positive Prompt
                exec_kwargs = {
                    "clip": clip,
                    "prompt": prompt,
                    "vae": vae
                }
                if img_comfy is not None:
                    exec_kwargs["image1"] = img_comfy
                
                pos_res = TextEncodeQwenImageEditPlus.execute(**exec_kwargs)
                positive = pos_res.result
                
                # Negative Prompt
                neg_res = TextEncodeQwenImageEditPlus.execute(
                    clip=clip, 
                    prompt="", 
                    vae=vae
                )
                negative = neg_res.result
                
                # Format conditioning
                if isinstance(positive, tuple) and len(positive) > 0: positive = positive[0]
                if isinstance(negative, tuple) and len(negative) > 0: negative = negative[0]
                if not isinstance(positive, list): positive = [positive]
                if not isinstance(negative, list): negative = [negative]
                
                # Empty Latent
                latent_res = empty_latent_node.generate(
                    width=target_width, 
                    height=target_height, 
                    batch_size=1
                )
                latent = latent_res[0]
                
                # KSampler
                actual_seed = seed if (seed is not None and seed != -1) else 42
                if actual_seed > 0xffffffffffffffff:
                    actual_seed = actual_seed % 0xffffffffffffffff
                
                samples_res = sampler_node.sample(
                    model=model,
                    seed=actual_seed,
                    steps=steps,
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    positive=positive,
                    negative=negative,
                    latent_image=latent,
                    denoise=1.0
                )
                samples = samples_res[0]
                
                # VAE Decode
                image_res = vae_decode_node.decode(
                    samples=samples, 
                    vae=vae
                )
                output_images = image_res[0]
                
                # Convert to bytes
                img_np = (output_images.squeeze().cpu().numpy() * 255).astype(np.uint8)
                output_pil = Image.fromarray(img_np)
                
                out_byte_arr = io.BytesIO()
                output_pil.save(out_byte_arr, format='PNG')
                result_bytes = out_byte_arr.getvalue()
                
                # Send result
                output_queue.put((request_id, {'status': 'success', 'data': result_bytes}))
                
                print(f"Worker {os.getpid()}: Finished request {request_id} in {time.time() - start_time:.2f}s")
                
                # Clear cache after each request to prevent fragmentation
                torch.cuda.empty_cache()
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                print(f"Worker {os.getpid()} error: {error_msg}")
                
                # 检查是否为显存溢出 (OOM)
                is_oom = "out of memory" in error_msg.lower() or "CUDA out of memory" in error_msg
                
                if is_oom:
                    print(f"Worker {os.getpid()} detected OOM, reporting and exiting for restart...")
                    output_queue.put((request_id, {'status': 'oom', 'message': error_msg}))
                    # 退出进程以便父进程重新启动
                    sys.exit(1)
                else:
                    output_queue.put((request_id, {'status': 'error', 'message': error_msg}))
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Worker {os.getpid()} failed to start: {e}")
