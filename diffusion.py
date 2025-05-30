import torch
import gc
import os
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

# Enable CUDA configurations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
gc.collect()

adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2", 
    torch_dtype=torch.float16  # Use float16 for GPU acceleration
)

model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffPipeline.from_pretrained(
    model_id, 
    motion_adapter=adapter, 
    torch_dtype=torch.float16  # Use float16 for GPU acceleration
)
scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe.scheduler = scheduler

pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()  # Uncomment to help with VRAM management

output = pipe(
    prompt=(
        "sóng gợn trường gian buồn điệp điệp"
    ),
    negative_prompt="vui vẻ, màu sắc tươi sáng, cảnh đông đúc, yếu tố hiện đại, chất lượng kém, chất lượng tệ hơn",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.Generator("cuda").manual_seed(42)
    # Remove device="cpu" to use default GPU
)
frames = output.frames[0]
export_to_gif(frames, "animation.gif")