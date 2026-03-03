import sys
sys.path.insert(0, '/tmp/ComfyUI')
import asyncio
from custom_nodes.ComfyUI_SeedVR2_VideoUpscaler.src.interfaces import comfy_entrypoint

async def test():
    try:
        ext = await comfy_entrypoint()
        print("Extension loaded successfully:", ext)
        nodes = await ext.get_node_list()
        print("Node list:", nodes)
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(test())
