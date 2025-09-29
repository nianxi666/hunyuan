import subprocess
import modal

# --- 配置 ---
VOLUME_NAME = "my-notebook-volume"
APP_NAME = "interactive-app"
APP_DIR = "/app"

# --- Modal App 设置 ---
app = modal.App(APP_NAME)

# 定义环境镜像
# [已修改] 这里是根据您指定的依赖更新后的镜像定义。
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "curl")
    # 对于需要特殊安装标记（如 --index-url）的库，使用 run_commands 更可靠
    .run_commands(
        # 安装指定 CUDA 版本的 PyTorch
        "pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128",
        # 安装 flash-attn 并禁用构建隔离
        "pip install flash-attn==2.8.3 --no-build-isolation",
    )
    # 安装其余的标准 Python 依赖
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
        # 以下是您原始脚本中的一些库，为了防止遗漏，我暂时保留了它们。
        # 如果确认不再需要，可以自行删除。
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
    gpu="B200",  # 注意：B200 GPU 需要 PyTorch 支持 CUDA 12.4+，这里的 cu128 是兼容的
    timeout=3600,
)
def run_command_in_container(command: str):
    """在 Modal 容器内执行指定的 shell 命令。"""
    print(f"准备执行命令: '{command}'")
    try:
        # 使用 subprocess.run 执行命令，并设置工作目录
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
