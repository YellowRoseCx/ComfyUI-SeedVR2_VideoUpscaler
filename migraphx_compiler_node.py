"""
Generic Torch Compile Node with MIGraphX Support for ComfyUI
"""

import torch
from comfy_api.latest import io
from functools import partial

# Conditionally import torch_migraphx
torch_mgx = False
try:
    import torch_migraphx
    from torch_migraphx.dynamo import migraphx_backend, migraphx_aot_backend
    print("Torch_MiGraphX available.")
    torch_mgx = True
except ImportError as e:
    print(f"Torch_MiGraphX not available: {e}")


class GenericTorchCompileMIGraphX(io.ComfyNode):
    """
    A standalone generic node to run torch.compile on a given model
    with specific support for MIGraphX backend configurations.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        backend_options = ["inductor", "cudagraphs"]
        if torch_mgx:
            backend_options.extend(["migraphx", "migraphx_aot"])

        return io.Schema(
            node_id="GenericTorchCompileMIGraphX",
            display_name="Torch Compile Model (MIGraphX)",
            category="Optimization",
            description=(
                "A generic node to wrap a model with torch.compile. "
                "Provides specialized configuration options for MIGraphX backend "
                "on ROCm platforms, alongside standard Inductor/CUDAGraphs backends."
            ),
            inputs=[
                io.Custom("MODEL").Input(
                    tooltip="The PyTorch model to compile."
                ),
                io.Combo.Input("backend",
                    options=backend_options,
                    default="migraphx_aot" if torch_mgx else "inductor",
                    tooltip=(
                        "Compilation backend:\n"
                        "• inductor: Full optimization with Triton kernel generation\n"
                        "• cudagraphs: Lightweight wrapper using CUDA graphs\n"
                        "• migraphx: Optimize with ROCm torch_migraphx backend\n"
                        "• migraphx_aot: Optimize with ROCm torch_migraphx backend (AOT compiled)"
                    )
                ),
                io.Boolean.Input("fullgraph",
                    default=False,
                    tooltip=(
                        "Compile entire model as single graph without breaks.\n"
                        "• False: Allow graph breaks (default)\n"
                        "• True: Enforce no breaks for maximum optimization"
                    )
                ),
                io.Boolean.Input("dynamic",
                    default=False,
                    tooltip=(
                        "Handle varying input shapes without recompilation.\n"
                        "• False: Specialize for exact input shapes (default)\n"
                        "• True: Create dynamic kernels that adapt to shape variations"
                    )
                ),
                io.Boolean.Input("mgx_fp16",
                    default=True,
                    tooltip="MIGraphX: fp16 quantize (default: True). Used if backend is migraphx or migraphx_aot."
                ),
                io.Boolean.Input("mgx_bf16",
                    default=False,
                    tooltip="MIGraphX: bf16 quantize (default: False). Used if backend is migraphx or migraphx_aot."
                ),
                io.Boolean.Input("mgx_exhaustive_tune",
                    default=False,
                    tooltip="MIGraphX: Perform exhaustive tune (default: False). Used if backend is migraphx or migraphx_aot."
                ),
                io.Boolean.Input("mgx_save_mxr",
                    default=False,
                    tooltip="MIGraphX: Save compiled MXR file (default: False). Used if backend is migraphx or migraphx_aot."
                ),
                io.Boolean.Input("mgx_deallocate",
                    default=True,
                    tooltip="MIGraphX: Enable memory deallocation (default: True). Used if backend is migraphx or migraphx_aot."
                ),
            ],
            outputs=[
                io.Custom("MODEL").Output(
                    tooltip="The torch.compiled model object."
                )
            ]
        )

    @classmethod
    def execute(cls, model: torch.nn.Module, backend: str, fullgraph: bool, dynamic: bool,
                mgx_fp16: bool, mgx_bf16: bool, mgx_exhaustive_tune: bool,
                mgx_save_mxr: bool, mgx_deallocate: bool) -> io.NodeOutput:
        """
        Compiles the incoming model using torch.compile with the specified configuration.
        """

        # Resolve backend callable if migraphx
        compile_backend = backend

        if torch_mgx and backend.startswith("migraphx"):
            # Set up partial backend for migraphx
            if backend == "migraphx":
                compile_backend = partial(
                    migraphx_backend,
                    fp16=mgx_fp16,
                    bf16=mgx_bf16,
                    exhaustive_tune=mgx_exhaustive_tune,
                    save_mxr=mgx_save_mxr,
                    deallocate=mgx_deallocate
                )
            elif backend == "migraphx_aot":
                compile_backend = partial(
                    migraphx_aot_backend,
                    fp16=mgx_fp16,
                    bf16=mgx_bf16,
                    exhaustive_tune=mgx_exhaustive_tune,
                    save_mxr=mgx_save_mxr,
                    deallocate=mgx_deallocate
                )

        print(f"Compiling model with backend: {backend}, fullgraph={fullgraph}, dynamic={dynamic}")

        # In case the backend doesn't take 'mode=None' cleanly, we omit mode for migraphx
        if torch_mgx and backend.startswith("migraphx"):
            compiled_model = torch.compile(
                model,
                backend=compile_backend,
                fullgraph=fullgraph,
                dynamic=dynamic
            )
        else:
            compiled_model = torch.compile(
                model,
                backend=compile_backend,
                fullgraph=fullgraph,
                dynamic=dynamic,
                mode="default"
            )

        return io.NodeOutput(compiled_model)
