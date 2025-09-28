import subprocess
import inferless
from pydantic import BaseModel, Field
from typing import Optional

@inferless.request
class RequestObjects(BaseModel):
    command: str = Field(default="echo Hello, Inferless!")

@inferless.response
class ResponseObjects(BaseModel):
    stdout: str = Field(default="")
    stderr: str = Field(default="")
    returncode: int = Field(default=0)

app = inferless.Cls(gpu="A100")  # 这里可以选择实际需要的 GPU 类型，或省略 GPU 参数

class TerminalController:
    @app.load
    def initialize(self):
        # 初始化时可以做环境准备等操作
        pass

    @app.infer
    def infer(self, request: RequestObjects) -> ResponseObjects:
        # 运行用户传入的终端命令
        process = subprocess.Popen(
            request.command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate()
        # 返回执行结果和日志
        return ResponseObjects(
            stdout=stdout.decode("utf-8"),
            stderr=stderr.decode("utf-8"),
            returncode=process.returncode
        )

    def finalize(self):
        # 清理资源
        pass

@inferless.local_entry_point
def my_local_entry(dynamic_params):
    controller = TerminalController()
    request_obj = RequestObjects(**dynamic_params)
    return controller.infer(request_obj)
