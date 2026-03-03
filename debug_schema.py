import sys
sys.path.insert(0, '/tmp/ComfyUI')
import asyncio
from custom_nodes.ComfyUI_SeedVR2_VideoUpscaler.src.interfaces import comfy_entrypoint

async def test():
    try:
        ext = await comfy_entrypoint()
        nodes = await ext.get_node_list()
        for n in nodes:
            try:
                schema = n.GET_SCHEMA()
                print(n, "schema loaded")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Failed to get schema for:", n)
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(test())
