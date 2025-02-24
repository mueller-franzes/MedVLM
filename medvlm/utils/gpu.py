import torch

def get_gpu_with_max_free_memory():
    if not torch.cuda.is_available():
        return None, "No CUDA-compatible GPU available."
    
    max_free_memory = 0
    best_gpu_index = -1
    
    for i in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{i}")
        # torch.cuda.set_device(device)
        try:
            free_memory, total_memory = torch.cuda.mem_get_info(device)
        except Exception:
            free_memory = -1 
        
        if free_memory > max_free_memory:
            max_free_memory = free_memory
            best_gpu_index = i
    
    if best_gpu_index == -1:
        return None, "No GPU detected."
    
    return best_gpu_index, max_free_memory