#!/usr/bin/env python
"""在能正常运行的 5080 环境执行，收集关键配置信息"""
import os
import sys
import json
import subprocess
from datetime import datetime

def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode().strip()
    except:
        return "N/A"

def get_torch_info():
    try:
        import torch
        return {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "compute_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
            "test_cuda": "✅" if torch.zeros(1, device='cuda').is_cuda else "❌"
        }
    except Exception as e:
        return {"error": str(e)}

def get_sapien_info():
    try:
        import sapien
        import mani_skill
        # 测试实际环境创建
        env = mani_skill.make(
            'PutOnPlateInScene25Main-v3',
            sim_backend='gpu',
            render_mode='none',
            num_envs=1
        )
        env.close()
        return {
            "sapien_version": sapien.__version__,
            "mani_skill_version": mani_skill.__version__,
            "env_creation": "✅ Success"
        }
    except Exception as e:
        return {"error": str(e)}

def get_system_info():
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "nvidia_driver": run_cmd("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"),
        "cuda_runtime": run_cmd("nvcc --version | grep release | cut -d',' -f2 | cut -d' ' -f3"),
        "pip_list_torch_related": run_cmd("pip list | grep -iE 'torch|sapien|mani|flash'"),
    }

def get_training_config():
    """尝试从项目配置中提取关键参数"""
    config_snippet = {}
    try:
        # 如果项目用 Hydra，尝试打印配置
        result = run_cmd("python examples/embodiment/train_embodied_agent.py --config-path examples/embodiment/config/ --config-name maniskill_ppo_openvla --cfg job 2>/dev/null | grep -A 20 'env:' | head -30")
        config_snippet["hydra_env_config"] = result[:500] + "..." if len(result) > 500 else result
    except:
        pass
    return config_snippet

if __name__ == "__main__":
    print(f"=== Environment Info Collector ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Hostname: {run_cmd('hostname')}")
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "hostname": run_cmd("hostname"),
        "torch": get_torch_info(),
        "sapien": get_sapien_info(),
        "system": get_system_info(),
        "config_snippet": get_training_config(),
    }
    
    # 打印可读格式
    print("\n=== TORCH ===")
    print(json.dumps(result["torch"], indent=2, default=str))
    
    print("\n=== SAPIEN/MANI_SKILL ===")
    print(json.dumps(result["sapien"], indent=2, default=str))
    
    print("\n=== SYSTEM ===")
    print(json.dumps(result["system"], indent=2, default=str))
    
    print("\n=== CONFIG SNIPPET ===")
    print(result["config_snippet"].get("hydra_env_config", "N/A"))
    
    # 保存 JSON 方便复制
    with open("/tmp/env_info_5080.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n💾 Saved to /tmp/env_info_5080.json")