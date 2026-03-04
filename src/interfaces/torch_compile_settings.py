"""
SeedVR2 Torch Compile Settings Node
Configure torch.compile optimization for DiT and VAE models
"""

from comfy_api.latest import io
from typing import Dict, Any, Tuple
from functools import partial

torch_mgx = False
try:
    import torch_migraphx
    from torch_migraphx.dynamo import migraphx_backend, migraphx_aot_backend
    print("Torch_MiGraphX available.")
    torch_mgx = True
except ImportError as e:
    print(f"Torch_MiGraphX not available: {e}")



class SeedVR2TorchCompileSettings(io.ComfyNode):
    """Configure torch.compile optimization for DiT and VAE models"""
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        backend_options = ["inductor", "cudagraphs"]
        if torch_mgx:
            backend_options.extend(["migraphx", "migraphx_aot"])

        return io.Schema(
            node_id="SeedVR2TorchCompileSettings",
            display_name="SeedVR2 Torch Compile Settings",
            category="SEEDVR2",
            description=(
                "Configure SeedVR2 torch.compile optimization for 20-40% DiT speedup and 15-25% VAE speedup. "
                "Trades longer first-run compilation time for faster inference.\n\n"
                "Connect to DiT and/or VAE model loaders. Requires PyTorch 2.0+ and Triton for inductor backend."
            ),
            inputs=[
                io.Combo.Input("backend", 
                    options=backend_options,
                    default="inductor",
                    tooltip=(
                        "Compilation backend:\n"
                        "• inductor: Full optimization with Triton kernel generation and fusion (recommended)\n"
                        "• cudagraphs: Lightweight wrapper using CUDA graphs, no kernel optimization\n"
                        "• migraphx: Optimize with ROCm torch_migraphx backend\n"
                        "• migraphx_aot: Optimize with ROCm torch_migraphx backend (AOT compiled)"
                    )
                ),
                io.Combo.Input("mode",
                    options=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                    default="default",
                    tooltip=(
                        "Optimization level (compilation time vs runtime performance):\n"
                        "• default: Fast compilation with good speedup (recommended for development)\n"
                        "• reduce-overhead: Lower overhead, optimized for smaller models\n"
                        "• max-autotune: Slowest compilation, best runtime performance (recommended for production)\n"
                        "• max-autotune-no-cudagraphs: Like max-autotune but without CUDA graphs"
                    )
                ),
                io.Boolean.Input("fullgraph",
                    default=False,
                    tooltip=(
                        "Compile entire model as single graph without breaks.\n"
                        "• False: Allow graph breaks for better compatibility (default)\n"
                        "• True: Enforce no breaks for maximum optimization (may fail with dynamic shapes)"
                    )
                ),
                io.Boolean.Input("dynamic",
                    default=False,
                    tooltip=(
                        "Handle varying input shapes without recompilation.\n"
                        "• False: Specialize for exact input shapes (default)\n"
                        "• True: Create dynamic kernels that adapt to shape variations\n"
                        "\n"
                        "Enable when processing different resolutions or batch sizes."
                    )
                ),
                io.Int.Input("dynamo_cache_size_limit",
                    default=64,
                    min=0,
                    max=1024,
                    step=1,
                    tooltip=(
                        "Maximum cached compiled versions per function (default: 64).\n"
                        "Controls how many shape variations to compile before stopping.\n"
                        "\n"
                        "• Increase: When processing many different input shapes (more memory usage)\n"
                        "• Decrease: When recompilation cost outweighs benefits (faster fallback to eager)"
                    )
                ),
                io.Int.Input("dynamo_recompile_limit",
                    default=128,
                    min=0,
                    max=1024,
                    step=1,
                    tooltip=(
                        "Maximum recompilation attempts before fallback to eager mode (default: 128).\n"
                        "Safety limit to prevent infinite compilation loops.\n"
                        "\n"
                        "Only increase if you see 'hit config.recompile_limit' warnings and have bounded shape variations."
                    )
                ),
                io.Boolean.Input("mgx_fp16",
                    default=True,
                    tooltip="MIGraphX parameter: fp16 quantize (default: True). Only used if backend is migraphx or migraphx_aot."
                ),
                io.Boolean.Input("mgx_bf16",
                    default=False,
                    tooltip="MIGraphX parameter: bf16 quantize (default: False). Only used if backend is migraphx or migraphx_aot."
                ),
                io.Boolean.Input("mgx_exhaustive_tune",
                    default=False,
                    tooltip="MIGraphX parameter: Perform exhaustive tune (default: False). Only used if backend is migraphx or migraphx_aot."
                ),
                io.Boolean.Input("mgx_save_mxr",
                    default=False,
                    tooltip="MIGraphX parameter: Save compiled MXR file (default: False). Only used if backend is migraphx or migraphx_aot."
                ),
                io.Boolean.Input("mgx_deallocate",
                    default=True,
                    tooltip="MIGraphX parameter: Enable memory deallocation (default: True). Only used if backend is migraphx or migraphx_aot."
                ),
            ],
            outputs=[
                io.Custom("TORCH_COMPILE_ARGS").Output(
                    tooltip="torch.compile optimization settings including backend, mode, and Dynamo configuration. Connect to DiT and/or VAE model loader nodes."
                )
            ]
        )
    
    @classmethod
    def execute(cls, backend: str, mode: str, fullgraph: bool, dynamic: bool, 
                   dynamo_cache_size_limit: int, dynamo_recompile_limit: int,
                   mgx_fp16: bool, mgx_bf16: bool, mgx_exhaustive_tune: bool,
                   mgx_save_mxr: bool, mgx_deallocate: bool) -> io.NodeOutput:
        """
        Create torch.compile configuration for model optimization
        
        Args:
            backend: Compilation backend ("inductor", "cudagraphs", "migraphx", etc.)
            mode: Optimization mode ("default", "reduce-overhead", "max-autotune", etc.)
            fullgraph: Whether to compile entire model as single graph
            dynamic: Whether to handle varying input shapes without recompilation
            dynamo_cache_size_limit: Maximum cached compiled versions per function
            dynamo_recompile_limit: Maximum recompilation attempts before fallback
            mgx_fp16: MIGraphX fp16 quantize
            mgx_bf16: MIGraphX bf16 quantize
            mgx_exhaustive_tune: MIGraphX exhaustive tune
            mgx_save_mxr: MIGraphX save mxr file
            mgx_deallocate: MIGraphX deallocate
            
        Returns:
            NodeOutput containing torch.compile configuration dictionary
        """
        if torch_mgx:
            if backend == "migraphx":
                mode = None
                backend = partial(migraphx_backend, fp16=mgx_fp16, bf16=mgx_bf16,
                                  exhaustive_tune=mgx_exhaustive_tune, save_mxr=mgx_save_mxr,
                                  deallocate=mgx_deallocate)
            elif backend == "migraphx_aot":
                mode = None
                backend = partial(migraphx_aot_backend, fp16=mgx_fp16, bf16=mgx_bf16,
                                  exhaustive_tune=mgx_exhaustive_tune, save_mxr=mgx_save_mxr,
                                  deallocate=mgx_deallocate)

        compile_args = {
            "backend": backend,
            "mode": mode,
            "fullgraph": fullgraph,
            "dynamic": dynamic,
            "dynamo_cache_size_limit": dynamo_cache_size_limit,
            "dynamo_recompile_limit": dynamo_recompile_limit,
        }
        return io.NodeOutput(compile_args)