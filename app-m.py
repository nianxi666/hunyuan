import subprocess
import modal

# --- Configuration ---
VOLUME_NAME = "my-notebook-volume"
APP_NAME = "interactive-app"
APP_DIR = "/app"

# --- Modal App Setup ---
app = modal.App(APP_NAME)

# Define the environment image
image = (
    # Start with a standard Modal image that has Python 3.12 configured correctly.
    modal.Image.debian_slim(python_version="3.12")
    # Install dependencies needed for adding the NVIDIA repository and building packages.
    .apt_install("git", "curl", "wget", "gnupg")
    .run_commands(
        "pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128",
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

# Create or get a persistent storage volume
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# --- Modal Function Definition ---
@app.function(
    image=image,
    volumes={APP_DIR: volume},
    gpu="B200",
    timeout=3600,
)
def run_command_in_container(command: str):
    """Executes a shell command inside the Modal container."""
    print(f"Preparing to execute command: '{command}'")
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
        print("--- Command Output ---")
        print(process.stdout)
        if process.stderr:
            print("--- Error Output ---")
            print(process.stderr)
        print("\nCommand executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\nCommand failed with return code: {e.returncode}")
        print("--- Command Output ---")
        print(e.stdout)
        print("--- Error Output ---")
        print(e.stderr)

# --- CLI Entrypoint ---
@app.local_entrypoint()
def main(command: str):
    """Local entrypoint to trigger the remote Modal function."""
    print(f"Remotely executing command via Modal: '{command}'")
    run_command_in_container.remote(command)
