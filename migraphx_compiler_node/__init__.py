from comfy_api.latest import ComfyExtension, io
from .migraphx_compiler_node import GenericTorchCompileMIGraphX

class MIGraphXCompilerExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [GenericTorchCompileMIGraphX]

async def comfy_entrypoint() -> ComfyExtension:
    return MIGraphXCompilerExtension()
