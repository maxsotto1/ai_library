import os
import pickle
import torch

def atomic_save(obj, final_path, use_pytorch=False):
    """
    Safely and atomically writes an object to disk.
    If use_pytorch=True, uses torch.save. Otherwise, uses pickle.
    """
    temp_path = final_path + ".tmp"
    try:
        if use_pytorch:
            # torch.save natively handles both state dicts and entire models
            torch.save(obj, temp_path)
        else:
            with open(temp_path, "wb") as f:
                pickle.dump(obj, f)
        
        # Flush the OS buffers to physical disk
        os.sync() 
        
        # Atomically swap the temp file with the target production path
        os.replace(temp_path, final_path)
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e