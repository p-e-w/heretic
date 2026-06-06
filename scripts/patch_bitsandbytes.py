import os
import shutil
import importlib.util
import re

def compile_bitsandbytes_dll(arch: str) -> str | None:
    import subprocess
    import tempfile
    
    # 1. Check if ROCm path and build tools are available
    rocm_path = os.environ.get("ROCM_PATH")
    if not rocm_path:
        # Check default paths
        for v in ["7.13.0", "7.12.0", "7.11.0", "7.10.0", "7.0.0", "6.3.0", "6.2.0", "6.1.0", "6.0.0"]:
            p = f"C:\\Program Files\\AMD\\ROCm\\{v}"
            if os.path.exists(p):
                rocm_path = p
                break
    if not rocm_path or not os.path.exists(rocm_path):
        return None

    # Check for cmake and ninja
    try:
        subprocess.check_call(["cmake", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call(["ninja", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return None

    # Locate vswhere or MSVC installation
    vswhere = "C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer\\vswhere.exe"
    vs_path = None
    if os.path.exists(vswhere):
        try:
            vs_path = subprocess.check_output([vswhere, "-latest", "-property", "installationPath"], text=True).strip()
        except Exception:
            pass
    if not vs_path:
        return None

    print(f"\n[INFO] Found AMD ROCm SDK at: {rocm_path}")
    print(f"[INFO] Found Visual Studio 2022 at: {vs_path}")
    
    # Prompt the user if they want to compile automatically
    print(f"\nWould you like Heretic to automatically clone the bitsandbytes repository and compile the native {arch} DLL for you?")
    try:
        import questionary
        from questionary import Choice
        choice = questionary.select(
            "Select action:",
            choices=[
                Choice("Yes, compile bitsandbytes for " + arch + " automatically", "compile"),
                Choice("No, skip compilation", "skip")
            ],
            style=questionary.Style([("highlighted", "reverse")]),
        ).ask()
    except Exception:
        # Fallback to stdin if questionary is not available
        try:
            choice = input("Yes, compile bitsandbytes for " + arch + " automatically? [y/N]: ").strip().lower()
            choice = "compile" if choice in ["y", "yes"] else "skip"
        except Exception:
            choice = "skip"

    if choice != "compile":
        return None

    # Execute build in temp directory
    temp_dir = tempfile.mkdtemp()
    try:
        print(f"Cloning bitsandbytes repository to: {temp_dir}...")
        subprocess.check_call(["git", "clone", "https://github.com/bitsandbytes-foundation/bitsandbytes.git", temp_dir])
        
        # We need to find vcvarsall.bat to run build under MSVC environment
        vcvarsall = os.path.join(vs_path, "VC", "Auxiliary", "Build", "vcvarsall.bat")
        if not os.path.exists(vcvarsall):
            print(f"Error: Could not find vcvarsall.bat at: {vcvarsall}")
            return None

        # Build command sequence
        # We run the build inside a batch script wrapper that sets up vcvarsall.bat first
        build_bat = os.path.join(temp_dir, "build_bnb.bat")
        with open(build_bat, "w", encoding="utf-8") as f:
            f.write(f'@echo off\n')
            f.write(f'call "{vcvarsall}" x64\n')
            f.write(f'set ROCM_PATH={rocm_path}\n')
            f.write(f'set HIP_PATH=%ROCM_PATH%\n')
            f.write(f'set PATH=%ROCM_PATH%\\bin;%ROCM_PATH%\\lib;%PATH%\n')
            f.write(f'cd /d "{temp_dir}"\n')
            f.write(f'cmake -G Ninja -B build -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="{arch}" -DCMAKE_BUILD_TYPE=Release\n')
            f.write(f'cmake --build build --config Release\n')

        print(f"Compiling bitsandbytes DLL using MSVC and Ninja...")
        subprocess.check_call([build_bat])
        
        # Locate built DLL
        built_dll = None
        for root, dirs, files in os.walk(os.path.join(temp_dir, "build")):
            for file in files:
                if file.endswith(".dll") and "bitsandbytes" in file:
                    built_dll = os.path.join(root, file)
                    break
        
        if built_dll and os.path.exists(built_dll):
            # Save to bin directory of heretic project
            repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            bin_dir = os.path.join(repo_dir, "bin")
            os.makedirs(bin_dir, exist_ok=True)
            dest_bin_dll = os.path.join(bin_dir, f"libbitsandbytes_rocm_{arch}.dll")
            shutil.copyfile(built_dll, dest_bin_dll)
            print(f"[SUCCESS] Successfully compiled and saved DLL to: {dest_bin_dll}")
            return dest_bin_dll
        else:
            print("Error: Compilation succeeded but the output DLL could not be located.")
            return None
    except Exception as e:
        print(f"Error: Compilation failed: {e}")
        return None
    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

def main():
    print("Locating bitsandbytes package...")
    spec = importlib.util.find_spec("bitsandbytes")
    if spec is None or spec.submodule_search_locations is None:
        print("Error: bitsandbytes is not installed in this environment.")
        return
    
    bnb_dir = spec.submodule_search_locations[0]
    print(f"Found bitsandbytes at: {bnb_dir}")
    
    # Try to detect GPU architecture
    arch = None
    try:
        import torch
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            if hasattr(prop, "gcnArchName"):
                arch = prop.gcnArchName.split(":")[0]
    except Exception:
        pass

    if not arch:
        # Fall back to wmic/powershell GPU name matching
        gpu_name = ""
        try:
            import subprocess
            gpu_name = subprocess.check_output("wmic path win32_VideoController get name", shell=True, text=True, stderr=subprocess.DEVNULL)
        except Exception:
            try:
                import subprocess
                gpu_name = subprocess.check_output('powershell -Command "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"', shell=True, text=True, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        
        if gpu_name:
            if re.search(r'\b(7900|7800|7700|7600|780M|760M|740M|W7900|W7800|W7700|W7600|W7500)\b', gpu_name, re.IGNORECASE):
                arch = "gfx1100"
            elif re.search(r'\b(9000|9070|9060|R9700|W8900|W8800|W8600)\b', gpu_name, re.IGNORECASE):
                arch = "gfx1200"
            elif any(brand in gpu_name for brand in ["AMD", "Radeon"]):
                arch = "gfx1030"

    if arch:
        print(f"Detected GPU architecture: {arch}")
    else:
        print("GPU architecture could not be determined. Defaulting to gfx1030.")
        arch = "gfx1030"

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Try architecture-specific DLL first
    dll_src = os.path.join(repo_dir, "bin", f"libbitsandbytes_rocm_{arch}.dll")
    if not os.path.exists(dll_src):
        dll_src = os.path.join(os.getcwd(), "bin", f"libbitsandbytes_rocm_{arch}.dll")
        
    if not os.path.exists(dll_src) and arch in ["gfx1100", "gfx1200"]:
        # Try to compile if SDK/tools are available
        dll_src = compile_bitsandbytes_dll(arch)
        if not dll_src:
            print(f"\n[WARNING] Precompiled DLL for {arch} was not found, and auto-compilation could not run.")
            print(f"[WARNING] 4-bit quantization on {arch.upper()} requires compiling bitsandbytes from source.")
            print(f"[WARNING] Please refer to WINDOWS_ROCM.md for manual compilation guides.")
            print(f"[WARNING] Falling back to RDNA2 (gfx1030) DLL, which may fail during kernel execution.\n")
            dll_src = None

    # Fall back to gfx1030
    if not dll_src or not os.path.exists(dll_src):
        print(f"Architecture-specific DLL for {arch} not found. Falling back to gfx1030...")
        dll_src = os.path.join(repo_dir, "bin", "libbitsandbytes_rocm_gfx1030.dll")
        if not os.path.exists(dll_src):
            dll_src = os.path.join(os.getcwd(), "bin", "libbitsandbytes_rocm_gfx1030.dll")

    # Fall back to original libbitsandbytes_rocm83.dll
    if not os.path.exists(dll_src):
        dll_src = os.path.join(repo_dir, "bin", "libbitsandbytes_rocm83.dll")
        if not os.path.exists(dll_src):
            dll_src = os.path.join(os.getcwd(), "bin", "libbitsandbytes_rocm83.dll")
        
    dll_dest = os.path.join(bnb_dir, "libbitsandbytes_rocm83.dll")
    
    if not os.path.exists(dll_src):
        print(f"Error: Precompiled DLL not found at: {dll_src}")
        return
    
    print(f"Copying precompiled ROCm DLL ({os.path.basename(dll_src)}) to: {dll_dest}")
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
