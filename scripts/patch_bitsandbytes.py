import os
import shutil
import importlib.util
import re

def main():
    print("Locating bitsandbytes package...")
    spec = importlib.util.find_spec("bitsandbytes")
    if spec is None or spec.submodule_search_locations is None:
        print("Error: bitsandbytes is not installed in this environment.")
        return
    
    bnb_dir = spec.submodule_search_locations[0]
    print(f"Found bitsandbytes at: {bnb_dir}")
    
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dll_src = os.path.join(repo_dir, "bin", "libbitsandbytes_rocm83.dll")
    if not os.path.exists(dll_src):
        # Fall back to current working directory
        dll_src = os.path.join(os.getcwd(), "bin", "libbitsandbytes_rocm83.dll")
        
    dll_dest = os.path.join(bnb_dir, "libbitsandbytes_rocm83.dll")
    
    if not os.path.exists(dll_src):
        print(f"Error: Precompiled DLL not found at: {dll_src}")
        return
    
    print(f"Copying precompiled ROCm DLL to: {dll_dest}")
    shutil.copyfile(dll_src, dll_dest)
    
    # 1. Patch cuda_specs.py
    cuda_specs_path = os.path.join(bnb_dir, "cuda_specs.py")
    if os.path.exists(cuda_specs_path):
        print(f"Patching: {cuda_specs_path}")
        with open(cuda_specs_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Patch get_rocm_gpu_arch
        old_arch = """def get_rocm_gpu_arch() -> str:
    \"\"\"Get ROCm GPU architecture.\"\"\"
    logger = logging.getLogger(__name__)
    try:
        if torch.version.hip:
            result = subprocess.run(["rocminfo"], capture_output=True, text=True)"""
        
        new_arch = """def get_rocm_gpu_arch() -> str:
    \"\"\"Get ROCm GPU architecture.\"\"\"
    logger = logging.getLogger(__name__)
    try:
        if torch.version.hip:
            if torch.cuda.is_available():
                prop = torch.cuda.get_device_properties(0)
                if hasattr(prop, "gcnArchName"):
                    arch = prop.gcnArchName.split(":")[0]
                    if arch.startswith("gfx"):
                        return arch
            result = subprocess.run(["rocminfo"], capture_output=True, text=True)"""
        
        # Patch get_rocm_warpsize
        old_warp = """def get_rocm_warpsize() -> int:
    \"\"\"Get ROCm warp size.\"\"\"
    logger = logging.getLogger(__name__)
    try:
        if torch.version.hip:
            result = subprocess.run(["rocminfo"], capture_output=True, text=True)"""
        
        new_warp = """def get_rocm_warpsize() -> int:
    \"\"\"Get ROCm warp size.\"\"\"
    logger = logging.getLogger(__name__)
    try:
        if torch.version.hip:
            if torch.cuda.is_available():
                prop = torch.cuda.get_device_properties(0)
                if hasattr(prop, "warp_size"):
                    return prop.warp_size
            result = subprocess.run(["rocminfo"], capture_output=True, text=True)"""
        
        if old_arch in content:
            content = content.replace(old_arch, new_arch)
        if old_warp in content:
            content = content.replace(old_warp, new_warp)
        
        with open(cuda_specs_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("Successfully patched cuda_specs.py")
    else:
        print(f"Warning: {cuda_specs_path} not found.")

    # 2. Patch cextension.py
    cextension_path = os.path.join(bnb_dir, "cextension.py")
    if os.path.exists(cextension_path):
        print(f"Patching: {cextension_path}")
        with open(cextension_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Patch get_cusparse
        old_cusparse = "lib.get_cusparse.restype = ct.c_void_p"
        new_cusparse = """if hasattr(lib, "get_cusparse"):
            lib.get_cusparse.restype = ct.c_void_p"""
        
        if old_cusparse in content:
            content = content.replace(old_cusparse, new_cusparse)
        
        # Patch DLL directory loading inside get_native_library()
        old_load = """    # Try to load the library - any errors will propagate up
    dll = ct.cdll.LoadLibrary(str(binary_path))"""
        
        new_load = """    if os.name == "nt" and torch.version.hip:
        # Try to locate the ROCm SDK DLLs and add them to the DLL search directories
        rocm_path = os.environ.get("ROCM_PATH")
        paths_to_check = []
        if rocm_path:
            paths_to_check.append(os.path.join(rocm_path, "bin"))
            paths_to_check.append(rocm_path)
        
        # Dynamically locate any _rocm_sdk* packages in the same site-packages directory
        try:
            site_packages = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if os.path.isdir(site_packages):
                for name in os.listdir(site_packages):
                    if name.lower().startswith("_rocm_sdk"):
                        bin_path = os.path.join(site_packages, name, "bin")
                        if os.path.isdir(bin_path):
                            paths_to_check.append(bin_path)
        except Exception:
            pass
        
        for path in paths_to_check:
            if os.path.isdir(path):
                try:
                    os.add_dll_directory(path)
                    logger.debug(f"Added ROCm DLL search path: {path}")
                except Exception as e:
                    logger.warning(f"Failed to add ROCm DLL search path {path}: {e}")

    # Try to load the library - any errors will propagate up
    dll = ct.cdll.LoadLibrary(str(binary_path))"""
        
        if old_load in content:
            content = content.replace(old_load, new_load)
            
        with open(cextension_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("Successfully patched cextension.py")
    else:
        print(f"Warning: {cextension_path} not found.")

    print("\nPatching complete! bitsandbytes ROCm support is now enabled on Windows.")

if __name__ == "__main__":
    main()
