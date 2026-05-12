"""
CUDA 13 library preloader for Kaggle environments.

Kaggle ships PyTorch built against CUDA 13, but the CUDA 13 runtime libraries
live in /usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib which isn't
on the dynamic linker's default search path. This module preloads them via
ctypes so bitsandbytes can find them.

Usage:
    # Must be called BEFORE importing torch, bitsandbytes, transformers
    from src.cuda_setup import preload_cuda_libs
    preload_cuda_libs()
"""
import ctypes
import glob
import os


CU13_LIB_DIR = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib"


def preload_cuda_libs(verbose: bool = True) -> int:
    """Preload all CUDA 13 .so.13 libraries from the cu13 package.
    
    Returns:
        Number of libraries successfully preloaded.
    """
    if not os.path.isdir(CU13_LIB_DIR):
        if verbose:
            print(f"⚠️  {CU13_LIB_DIR} not found — skipping preload")
            print("    (this is expected outside Kaggle CUDA 13 environments)")
        return 0
    
    lib_paths = sorted(glob.glob(f"{CU13_LIB_DIR}/*.so.13"))
    loaded = 0
    
    for path in lib_paths:
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            loaded += 1
        except OSError as e:
            if verbose:
                print(f"⚠️  Could not preload {os.path.basename(path)}: {e}")
    
    if verbose:
        print(f"✅ Preloaded {loaded}/{len(lib_paths)} CUDA 13 libraries from {CU13_LIB_DIR}")
    
    return loaded


if __name__ == "__main__":
    preload_cuda_libs()