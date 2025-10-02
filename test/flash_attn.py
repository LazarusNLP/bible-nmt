import torch
print(torch.__version__)
print(torch.version.cuda)
# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")

import flash_attn_2_cuda as flash_attn_gpu