import subprocess
import modal

# --- 配置 ---
VOLUME_NAME = "my-notebook-volume"
APP_NAME = "interactive-app"
APP_DIR = "/app"

# --- Modal App 设置 ---
app = modal.App(APP_NAME)

# 定义环境镜像
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "curl")
    .run_commands(
        "pip install packaging",
        "pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128",
        # [已修正] 在编译前指定 CUDA_HOME 环境变量
        "CUDA_HOME=/usr/local/cuda pip install flash-attn==2.8.3 --no-build-isolation",
    )
    .pip_install(
        "einops>=0.8.0",
        "numpy==1.26.4",
        "pillow==11.3.0",
        "diffusers>=0.32.0",
        "safetensors==0.4.5",
        "tokenizers>=0.21.0",
        "transformers[accelerate,tiktoken]>=4.56.0",
        "huggingface_hub[cli]",
        "flashinfer-python",
        "sentencepiece",
        "dashscope",
        "peft",
        "requests",
    )
)

# 从统一名称创建或获取持久化存储卷
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# --- Modal 函数定义 ---
@app.function(
    image=image,
    volumes={APP_DIR: volume},
    gpu="B200",
    timeout=3600,
)
def run_command_in_container(command: str):
    """在 Modal 容器内执行指定的 shell 命令。"""
    print(f"准备执行命令: '{command}'")
    try:
        process = subprocess.run(
            command,
            shell=True,
            check=True,
            cwd=APP_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print("--- 命令输出 ---")
        print(process.stdout)
        if process.stderr:
            print("--- 错误输出 ---")
            print(process.stderr)
        print("\n命令执行成功。")
    except subprocess.CalledProcessError as e:
        print(f"\n命令执行失败，返回码: {e.returncode}")
        print("--- 命令输出 ---")
        print(e.stdout)
        print("--- 错误输出 ---")
        print(e.stderr)

# --- CLI 入口点 ---
@app.local_entrypoint()
def main(command: str):
    """本地入口点，用于触发远程 Modal 函数执行命令。"""
    print(f"正在通过 Modal 远程执行命令: '{command}'")
    run_command_in_container.remote(command)
