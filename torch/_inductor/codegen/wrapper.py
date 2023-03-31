import collections
import contextlib
import dataclasses
import functools
import hashlib
import os
from itertools import count
from typing import Any, Dict, List, Tuple

import sympy
from sympy import Expr

from torch._dynamo.utils import dynamo_timed
from .. import codecache, config, ir
from ..codecache import cubin_cache_dir
from ..utils import (
    cache_on_self,
    get_benchmark_name,
    has_triton,
    LineContext,
    sympy_dot,
    sympy_product,
    sympy_symbol,
)
from ..virtualized import V
from .common import CodeGen, DeferredLine, IndentedBuffer, Kernel, PythonPrinter


pexpr = PythonPrinter().doprint


def buffer_reuse_key(node: ir.Buffer):
    size = node.get_size()
    stride = node.get_stride()
    last_element = sympy_dot([s - 1 for s in size], stride)
    return (
        node.get_device(),
        node.get_dtype(),
        V.graph.sizevars.simplify(sympy_product(size)),
        # Detect gaps in tensor storage caused by strides
        V.graph.sizevars.size_hint(last_element),
    )


def is_int(s: str):
    try:
        int(s)
    except ValueError:
        return False
    return True


def is_float(s: str):
    try:
        float(s)
    except ValueError:
        return False
    return True


class KernelParamCache:
    cache = dict()

    def __init__(self):
        self.prev_cache = None

    def __enter__(self):
        self.prev_cache = KernelParamCache.cache
        KernelParamCache.cache = dict()

    def __exit__(self, *args):
        KernelParamCache.cache.clear()
        KernelParamCache.cache = self.prev_cache


class MemoryPlanningState:
    def __init__(self):
        super().__init__()
        self.reuse_pool: Dict[
            Any, List["FreeIfNotReusedLine"]
        ] = collections.defaultdict(list)

    def __contains__(self, key):
        return bool(self.reuse_pool.get(key, None))

    def pop(self, key) -> "FreeIfNotReusedLine":
        item = self.reuse_pool[key].pop()
        assert not item.is_reused
        return item

    def push(self, key, item: "FreeIfNotReusedLine"):
        assert not item.is_reused
        self.reuse_pool[key].append(item)


@dataclasses.dataclass
class EnterCudaDeviceContextManagerLine:
    device_idx: int

    def codegen(self, code: IndentedBuffer, device_cm_stack: contextlib.ExitStack):
        if V.graph.cpp_wrapper:
            code.writeline("\n")
            code.writeline(f"at::cuda::CUDAGuard device_guard({self.device_idx});")
        else:
            # Note _DeviceGuard has less overhead than device, but only accepts
            # integers
            code.writeline(f"with torch.cuda._DeviceGuard({self.device_idx}):")
            device_cm_stack.enter_context(code.indent())
            code.writeline(
                f"torch.cuda.set_device({self.device_idx}) # no-op to ensure context"
            )


class ExitCudaDeviceContextManagerLine:
    def codegen(self, code: IndentedBuffer, device_cm_stack: contextlib.ExitStack):
        if not V.graph.cpp_wrapper:
            device_cm_stack.close()


@dataclasses.dataclass
class MemoryPlanningLine:
    wrapper: "WrapperCodeGen"

    def plan(self, state: MemoryPlanningState) -> "MemoryPlanningLine":
        """First pass to find reuse"""
        return self

    def codegen(self, code: IndentedBuffer):
        """Second pass to output code"""
        pass


@dataclasses.dataclass
class AllocateLine(MemoryPlanningLine):
    node: ir.Buffer

    def plan(self, state: MemoryPlanningState):
        if self.node.get_name() in V.graph.removed_buffers:
            return NullLine(self.wrapper)

        # try to reuse a recently freed buffer
        key = buffer_reuse_key(self.node)
        if key in state:
            free_line = state.pop(key)
            free_line.is_reused = True
            return ReuseLine(self.wrapper, free_line.node, self.node)

        return self

    def codegen(self, code: IndentedBuffer):
        assert self.node.get_name() not in V.graph.removed_buffers
        line = self.wrapper.make_buffer_allocation(self.node)
        code.writeline(line)


@dataclasses.dataclass
class FreeIfNotReusedLine(MemoryPlanningLine):
    node: ir.Buffer
    is_reused: bool = False

    def plan(self, state: MemoryPlanningState):
        assert not self.is_reused
        if self.node.get_name() in V.graph.removed_buffers:
            return NullLine(self.wrapper)
        state.push(buffer_reuse_key(self.node), self)
        return self

    def codegen(self, code: IndentedBuffer):
        assert self.node.get_name() not in V.graph.removed_buffers
        if not self.is_reused:
            code.writeline(self.wrapper.make_buffer_free(self.node))


@dataclasses.dataclass
class ReuseLine(MemoryPlanningLine):
    node: ir.Buffer
    reused_as: ir.Buffer

    def plan(self, state: MemoryPlanningState):
        assert self.node.get_name() not in V.graph.removed_buffers
        assert self.reused_as.get_name() not in V.graph.removed_buffers
        return self

    def codegen(self, code: IndentedBuffer):
        assert self.node.get_name() not in V.graph.removed_buffers
        assert self.reused_as.get_name() not in V.graph.removed_buffers
        code.writeline(
            self.wrapper.make_buffer_reuse(
                self.node,
                self.reused_as,
            )
        )


class NullLine(MemoryPlanningLine):
    pass


class WrapperCodeGen(CodeGen):
    """
    Generate outer wrapper in Python that calls the kernels.
    """

    def __init__(self):
        super().__init__()
        self._names_iter = count()
        self.header = IndentedBuffer()
        self.prefix = IndentedBuffer()
        self.wrapper_call = IndentedBuffer()
        self.kernels = {}
        self.lines = []
        self.need_seed = False
        self.declare = ""
        self.ending = ""
        self.comment = "#"
        self.namespace = ""

        self.write_header()
        self.write_prefix()

        for name, value in V.graph.constants.items():
            # include a hash so our code cache gives different constants different files
            hashed = hashlib.sha256(repr(value).encode("utf-8")).hexdigest()
            self.header.writeline(f"{name} = None  # {hashed}")

        self.allocated = set()
        self.freed = set()

        # maps from reusing buffer to reused buffer
        self.reuses = dict()

        self.write_get_cuda_stream = functools.lru_cache(None)(
            self.write_get_cuda_stream
        )

        @functools.lru_cache(None)
        def add_import_once(line):
            self.header.writeline(line)

        self.add_import_once = add_import_once
        self._metas = {}

    def write_header(self):
        self.header.splice(
            f"""
                from ctypes import c_void_p, c_long
                import torch
                import math
                import random
                import os
                import tempfile
                from torch._inductor.utils import maybe_profile

                from torch import empty_strided, as_strided, device
                from {codecache.__name__} import AsyncCompile
                from torch._inductor.select_algorithm import extern_kernels

                aten = torch.ops.aten
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride
                async_compile = AsyncCompile()

            """
        )

        if has_triton():
            self.header.splice(
                """
                import triton
                import triton.language as tl
                from torch._inductor.triton_heuristics import grid, start_graph, end_graph
                from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
                """
            )

    def add_meta_once(self, meta):
        meta = repr(meta)
        if meta not in self._metas:
            var = f"meta{len(self._metas)}"
            self._metas[meta] = var
            self.header.writeline(f"{var} = {meta}")
        return self._metas[meta]

    @cache_on_self
    def get_output_refs(self):
        return [x.codegen_reference() for x in V.graph.graph_outputs]

    def write_prefix(self):
        self.prefix.splice(
            """

            async_compile.wait(globals())
            del async_compile

            def call(args):
            """
        )
        with self.prefix.indent():
            if config.triton.debug_sync_graph:
                self.prefix.writeline("torch.cuda.synchronize()")
            inp_len = len(V.graph.graph_inputs.keys())
            if inp_len != 0:
                lhs = f"{', '.join(V.graph.graph_inputs.keys())}{'' if inp_len != 1 else ','}"
                self.prefix.writeline(f"{lhs} = args")
                self.prefix.writeline("args.clear()")
            for name in V.graph.randomness_seeds:
                self.prefix.writeline(
                    f"torch.randint(2**31, size=(), dtype=torch.int64, out={name})"
                )
            self.codegen_inputs(self.prefix, V.graph.graph_inputs)

    def append_precomputed_sizes_to_prefix(self):
        with self.prefix.indent():
            self.codegen_precomputed_sizes(self.prefix)

    def write_get_cuda_stream(self, index):
        name = f"stream{index}"
        self.writeline(f"{name} = get_cuda_stream({index})")
        return name

    def next_kernel_suffix(self):
        return f"{next(self._names_iter)}"

    def codegen_cuda_device_guard_enter(self, device_idx):
        self.writeline(EnterCudaDeviceContextManagerLine(device_idx))

    def codegen_cuda_device_guard_exit(self):
        self.writeline(ExitCudaDeviceContextManagerLine())

    def generate_return(self, output_refs):
        if output_refs:
            self.wrapper_call.writeline("return (" + ", ".join(output_refs) + ", )")
        else:
            self.wrapper_call.writeline("return ()")

    def generate_end(self, result):
        return

    def generate_extern_kernel_out(
        self, output_view, codegen_reference, args, kernel, cpp_kernel
    ):
        if output_view:
            args.append(f"out={output_view.codegen_reference()}")
        else:
            args.append(f"out={codegen_reference}")
        self.writeline(f"{kernel}({', '.join(args)})")

    @dynamo_timed
    def generate(self):
        result = IndentedBuffer()
        result.splice(self.header)

        out_names = V.graph.get_output_names()
        with contextlib.ExitStack() as stack:
            stack.enter_context(self.wrapper_call.indent())
            if config.profiler_mark_wrapper_call:
                self.generate_profiler_mark_wrapper_call(stack)
            if config.profile_bandwidth:
                self.wrapper_call.writeline("start_graph()")

            while (
                self.lines
                and isinstance(self.lines[-1], MemoryPlanningLine)
                and self.lines[-1].node.name not in out_names
            ):
                # these lines will be pointless
                self.lines.pop()

            # codegen allocations in two passes
            planning_state = MemoryPlanningState()
            for i in range(len(self.lines)):
                if isinstance(self.lines[i], MemoryPlanningLine):
                    self.lines[i] = self.lines[i].plan(planning_state)

            device_cm_stack = contextlib.ExitStack()
            for line in self.lines:
                if isinstance(line, MemoryPlanningLine):
                    line.codegen(self.wrapper_call)
                elif isinstance(
                    line,
                    (
                        EnterCudaDeviceContextManagerLine,
                        ExitCudaDeviceContextManagerLine,
                    ),
                ):
                    line.codegen(self.wrapper_call, device_cm_stack)
                else:
                    self.wrapper_call.writeline(line)

            output_refs = self.get_output_refs()
            if config.triton.debug_sync_graph:
                self.wrapper_call.writeline("torch.cuda.synchronize()")

            if config.profile_bandwidth:
                self.wrapper_call.writeline("end_graph()")

            self.generate_return(output_refs)

        self.append_precomputed_sizes_to_prefix()
        result.splice(self.prefix)

        with result.indent():
            result.splice(self.wrapper_call)

        self.generate_end(result)

        self.add_benchmark_harness(result)

        return result.getvaluewithlinemap()

    def codegen_inputs(self, code: IndentedBuffer, graph_inputs: Dict[str, ir.Buffer]):
        """Assign all symbolic shapes to locals"""
        if self.need_seed:
            code.writeline(
                "seed = torch.randint(2**31, size=(), dtype=torch.int32).item()"
            )

        @functools.lru_cache(None)
        def sizeof(name):
            code.writeline(f"{self.declare}{name}_size = {name}.size(){self.ending}")
            return f"{name}_size"

        @functools.lru_cache(None)
        def strideof(name):
            code.writeline(
                f"{self.declare}{name}_stride = {name}.stride(){self.ending}"
            )
            return f"{name}_stride"

        # Assign all symbolic shapes needed to local variables
        needed = set(V.graph.sizevars.var_to_val.keys()) - set(
            V.graph.sizevars.replacements.keys()
        )

        def is_expr(x):
            return isinstance(x[1], sympy.Expr)

        graph_inputs_expr = list(filter(is_expr, graph_inputs.items()))
        graph_inputs_tensors = list(
            filter(lambda x: not is_expr(x), graph_inputs.items())
        )

        for name, shape in graph_inputs_expr:
            shape = V.graph.sizevars.simplify(shape)
            if shape in needed:
                needed.remove(shape)
                code.writeline(f"{self.declare}{shape} = {name}{self.ending}")

        for name, value in graph_inputs_tensors:
            shapes = value.get_size()
            for dim, shape in enumerate(shapes):
                shape = V.graph.sizevars.simplify(shape)
                if shape in needed:
                    needed.remove(shape)
                    code.writeline(
                        f"{self.declare}{shape} = {sizeof(name)}[{dim}]{self.ending}"
                    )

        for name, value in graph_inputs_tensors:
            shapes = value.get_stride()
            for dim, shape in enumerate(shapes):
                shape = V.graph.sizevars.simplify(shape)
                if shape in needed:
                    needed.remove(shape)
                    code.writeline(
                        f"{self.declare}{shape} = {strideof(name)}[{dim}]{self.ending}"
                    )

    def codegen_precomputed_sizes(self, code: IndentedBuffer):
        for sym, expr in V.graph.sizevars.inv_precomputed_replacements.items():
            code.writeline(f"{self.declare}{sym} = {pexpr(expr)}")

    def codegen_sizevar(self, x: Expr) -> str:
        return pexpr(V.graph.sizevars.simplify(x))

    def codegen_shape_tuple(self, shape: Tuple[Expr, ...]) -> str:
        parts = list(map(self.codegen_sizevar, shape))
        if len(parts) == 0:
            return "()"
        if len(parts) == 1:
            return f"({parts[0]}, )"
        return f"({', '.join(parts)})"

    def benchmark_compiled_module(self, output):
        def add_fake_input(name, shape, stride, device, dtype):
            output.writeline(
                f"{name} = rand_strided("
                f"{self.codegen_shape_tuple(shape)}, "
                f"{self.codegen_shape_tuple(stride)}, "
                f"device='{device}', dtype={dtype})"
            )

        def add_expr_input(name, val):
            output.writeline(f"{name} = {val}")

        output.writelines(["", "", "def benchmark_compiled_module():"])
        with output.indent():
            output.splice(
                """
                from torch._dynamo.testing import rand_strided
                from torch._inductor.utils import print_performance
                """,
                strip=True,
            )

            for name, value in V.graph.constants.items():
                # all the constants are global variables, that's why we need
                # these 'global var_name' lines
                output.writeline(f"global {name}")
                add_fake_input(
                    name, value.size(), value.stride(), value.device, value.dtype
                )

            for name, value in V.graph.graph_inputs.items():
                if isinstance(value, sympy.Expr):  # Don't need to add symbolic
                    add_expr_input(name, V.graph.sizevars.size_hint(value))
                else:
                    shape = [V.graph.sizevars.size_hint(x) for x in value.get_size()]
                    stride = [V.graph.sizevars.size_hint(x) for x in value.get_stride()]
                    add_fake_input(
                        name, shape, stride, value.get_device(), value.get_dtype()
                    )

            call_str = f"call([{', '.join(V.graph.graph_inputs.keys())}])"
            output.writeline(f"print_performance(lambda: {call_str})")

    def add_benchmark_harness(self, output):
        """
        Append a benchmark harness to generated code for debugging
        """
        if not config.benchmark_harness:
            return

        self.benchmark_compiled_module(output)

        output.writelines(["", "", 'if __name__ == "__main__":'])
        with output.indent():
            output.writelines(
                [
                    "import argparse",
                    "from torch._inductor.utils import benchmark_all_kernels",
                    "",
                    "parser = argparse.ArgumentParser()",
                    'parser.add_argument("--benchmark-kernels", "-k", action="store_true", help="Whether to benchmark each individual kernels")',  # noqa: B950, line too long
                    'parser.add_argument("--benchmark-all-configs", "-c", action="store_true", help="Whether to benchmark each individual config for a kernel")',  # noqa: B950, line too long
                    'parser.add_argument("--profile", "-p", action="store_true", help="Whether to profile the compiled module")',  # noqa: B950, line too long
                    "args = parser.parse_args()",
                    "",
                    "if args.benchmark_kernels:",
                ]
            )
            with output.indent():
                output.writeline(
                    f"benchmark_all_kernels('{get_benchmark_name()}', args.benchmark_all_configs)"
                )
            output.writeline("else:")
            with output.indent():
                output.writeline("with maybe_profile(args.profile) as p:")
                with output.indent():
                    output.writeline("benchmark_compiled_module()")
                output.writeline("")
                output.writeline("if p:")
                with output.indent():
                    output.writelines(
                        [
                            'path = f"{tempfile.gettempdir()}/compiled_module_profile.json"',
                            "p.export_chrome_trace(path)",
                            'print(f"Chrome trace for the profile is written to {path}")',
                        ]
                    )

    def define_kernel(self, name: str, kernel: str, metadata: str = None):
        metadata_comment = f"{metadata}\n" if metadata else ""
        self.header.splice(f"\n\n{metadata_comment}{name} = {kernel}")

    def load_kernel(self, name: str = None, kernel: str = None, arg_types: List = None):
        return

    def wrap_kernel_call(self, name, call_args):
        return "{}({})".format(name, ", ".join(call_args))

    def generate_profiler_mark_wrapper_call(self, stack):
        self.wrapper_call.writeline("from torch.profiler import record_function")
        self.wrapper_call.writeline("with record_function('inductor_wrapper_call'):")
        stack.enter_context(self.wrapper_call.indent())

    def generate_kernel_call(self, name, call_args, device_index=None):
        self.writeline(self.wrap_kernel_call(name, call_args))

    def call_kernel(self, name: str, kernel: Kernel):
        tmp = IndentedBuffer()
        kernel.call_kernel(self, tmp, name)
        for line in tmp.getvalue().split("\n"):
            line = line.strip()
            if line:
                self.writeline(line)

    def writeline(self, line):
        self.lines.append(line)

    def enter_context(self, ctx):
        self.lines.append(LineContext(ctx))

    # The following methods are for memory management
    def make_buffer_allocation(self, buffer):
        device = buffer.get_device()
        dtype = buffer.get_dtype()
        shape = tuple(buffer.get_size())
        stride = tuple(buffer.get_stride())
        return (
            f"{buffer.get_name()} = empty_strided("
            f"{self.codegen_shape_tuple(shape)}, "
            f"{self.codegen_shape_tuple(stride)}, "
            f"device='{device.type}', dtype={dtype})"
        )

    def make_buffer_free(self, buffer):
        return f"del {buffer.get_name()}"

    def make_buffer_reuse(self, old, new):
        assert old.get_dtype() == new.get_dtype()
        del_line = ""
        if old.get_name() not in V.graph.get_output_names():
            del_line = f"; {self.make_buffer_free(old)}"
        if old.get_size() == new.get_size() and old.get_stride() == new.get_stride():
            return f"{self.declare}{new.get_name()} = {old.get_name()}{del_line}  {self.comment} reuse"

        return (
            f"{self.declare}{new.get_name()} = {self.namespace}as_strided({old.get_name()}, "
            f"{self.codegen_shape_tuple(new.get_size())}, "
            f"{self.codegen_shape_tuple(new.get_stride())}){del_line}  {self.comment} reuse"
        )

    def codegen_deferred_allocation(self, name, layout):
        self.writeline(
            DeferredLine(
                name,
                f"{self.declare}{name} = {layout.view.codegen_reference()}{self.ending}  {self.comment} alias",
            )
        )

    def codegen_allocation(self, buffer):
        name = buffer.get_name()
        if name in V.graph.removed_buffers or name in self.allocated:
            return
        self.allocated.add(name)
        if isinstance(
            buffer,
            (ir.ExternKernelAlloc, ir.MultiOutput),
        ):
            return

        layout = buffer.get_layout()
        if isinstance(layout, ir.MutationLayout):
            return
        if isinstance(layout, ir.AliasedLayout):
            assert isinstance(layout.view, ir.ReinterpretView)
            if not layout.maybe_guard_aligned():
                V.graph.unaligned_buffers.add(name)
            self.codegen_allocation(layout.view.data)
            self.codegen_deferred_allocation(name, layout)
            return

        self.writeline(AllocateLine(self, buffer))

    def codegen_free(self, buffer):
        name = buffer.get_name()

        # can be freed but not reused
        if isinstance(buffer, ir.InputBuffer):
            self.writeline(self.make_buffer_free(buffer))
            return

        if not self.can_reuse(buffer):
            return
        self.freed.add(name)

        layout = buffer.get_layout()
        if isinstance(layout, (ir.AliasedLayout, ir.MultiOutputLayout)):
            self.writeline(self.make_buffer_free(buffer))
            return

        self.writeline(FreeIfNotReusedLine(self, buffer))

    def can_reuse(self, buffer):
        name = buffer.get_name()
        if (
            name in V.graph.removed_buffers
            or name in V.graph.graph_inputs
            or name in V.graph.constants
            or name in self.freed
        ):
            return False
        return True

    def did_reuse(self, buffer, reused_buffer):
        # Check whether a given buffer was reused by a possible reuser in the wrapper codegen
        # Can be consulted from inside ir codegen, e.g. to determine whether a copy is needed
        return (
            buffer.get_name() in self.reuses
            and self.reuses[buffer.get_name()] == reused_buffer.get_name()
        )

    def codegen_inplace_reuse(self, input_buffer, output_buffer):
        assert buffer_reuse_key(input_buffer) == buffer_reuse_key(output_buffer)
        self.codegen_allocation(input_buffer)
        self.freed.add(input_buffer.get_name())
        self.allocated.add(output_buffer.get_name())
        self.reuses[output_buffer.get_name()] = input_buffer.get_name()
        self.writeline(ReuseLine(self, input_buffer, output_buffer))


class CppWrapperCodeGen(WrapperCodeGen):
    """
    Generates cpp wrapper for running on CPU and calls cpp kernels
    """

    def __init__(self):
        super().__init__()
        self.declare = "auto "
        self.ending = ";"
        self.comment = "//"
        self.namespace = "at::"

    def seed(self):
        """
        Seed is a special variable used to hold the rng seed for a graph.

        Note this is only used by the CPU backend, we put seeds in a
        1-element tensor for the CUDA backend.
        """
        self.need_seed = True
        return sympy_symbol("seed")

    def write_header(self):
        if V.graph.aot_mode:
            self.header.splice("\n#include <ATen/ATen.h>")
        else:
            self.header.splice(
                """
                from torch.utils.cpp_extension import load_inline

                cpp_wrapper_src = (
                '''
                """
            )

    @cache_on_self
    def get_output_refs(self):
        from ..ir import NoneAsConstantBuffer

        return [
            "at::Tensor()"
            if isinstance(x, NoneAsConstantBuffer)
            else x.codegen_reference()
            for x in V.graph.graph_outputs
        ]

    def call_func_name(self):
        return "inductor_cpp_entry"

    def write_prefix(self):
        return

    def write_wrapper_decl(self):
        inputs_len = len(V.graph.graph_inputs.keys())
        self.prefix.splice(
            f"""std::vector<at::Tensor> {self.call_func_name()}(const std::vector<at::Tensor>& args) {{"""
        )
        with self.wrapper_call.indent():
            if inputs_len != 0:
                inputs_keys_str = ", ".join(V.graph.graph_inputs.keys())
                self.wrapper_call.writeline(f"at::Tensor {inputs_keys_str};")
                for idx, input_key in enumerate(V.graph.graph_inputs.keys()):
                    self.wrapper_call.writeline(f"{input_key} = args[{idx}];")

            for name in V.graph.randomness_seeds:
                self.wrapper_call.writeline(f"at::Tensor {name};")
                self.wrapper_call.writeline(
                    f"{name} = at::randint(std::pow(2, 31), {{}}, at::ScalarType::Long);"
                )
            self.codegen_inputs(self.wrapper_call, V.graph.graph_inputs)

    def generate(self):
        self.write_wrapper_decl()
        return super().generate()

    def define_kernel(self, name: str, kernel: str, kernel_path: str = None):
        self.header.splice(f"\n{kernel}\n")

    def wrap_kernel_call(self, name, call_args):
        return f"{name}({', '.join(call_args)});"

    def generate_return(self, output_refs):
        self.wrapper_call.writeline(f"return {{{', '.join(output_refs)}}};\n}}")

    def generate_end(self, result):
        if not V.graph.aot_mode:
            result.writeline("'''\n)")
            # Generate load_inline to jit compile the generated cpp code and to use it in Python
            shared = codecache.get_shared()
            warning_all_flag = codecache.get_warning_all_flag()
            cpp_flags = codecache.cpp_flags()
            ipaths, lpaths, libs, macros = codecache.get_include_and_linking_paths(
                vec_isa=codecache.pick_vec_isa()
            )
            optimization_flags = codecache.optimization_flags()
            use_custom_generated_macros = codecache.use_custom_generated_macros()

            extra_cflags = f"{cpp_flags} {optimization_flags} {warning_all_flag} {macros} {use_custom_generated_macros}"
            extra_ldflags = f"{shared} {lpaths} {libs}"
            extra_include_paths = f"{ipaths}"

            # get the hash of the wrapper code to name the extension
            wrapper_call_hash = codecache.code_hash(self.wrapper_call.getvalue())
            result.splice(
                f"""
                module = load_inline(
                    name='cpp_wrapper_{wrapper_call_hash}',
                    cpp_sources=[cpp_wrapper_src],
                    functions=['inductor_cpp_entry'],
                    extra_cflags=['{extra_cflags}'],
                    extra_ldflags=['{extra_ldflags}'],
                    extra_include_paths=['{extra_include_paths}'])

                # Wrap the func to support setting result._boxed_call = True
                def wrap_func(f):
                    def g(args):
                        return f(args)
                    return g

                call = wrap_func(module.inductor_cpp_entry)
                """
            )

    def generate_extern_kernel_out(
        self, output_view, codegen_reference, args, kernel, cpp_kernel
    ):
        if output_view:
            output_as_strided = f"{output_view.codegen_reference()}"
            output_name = f"{output_view.get_name()}_as_strided"
            self.writeline(f"auto {output_name} = {output_as_strided};")

            args.insert(0, output_name)
        else:
            args.insert(0, f"{codegen_reference}")
        self.writeline(f"{cpp_kernel}({', '.join(args)});")

    def codegen_shape_tuple(self, shape: Tuple[Expr, ...]) -> str:
        parts = list(map(self.codegen_sizevar, shape))
        if len(parts) == 0:
            return "{}"
        if len(parts) == 1:
            return f"{{{parts[0]}, }}"
        return f"{{{', '.join(parts)}}}"

    def make_buffer_free(self, buffer):
        return f"{buffer.get_name()}.reset();"

    def generate_profiler_mark_wrapper_call(self, stack):
        self.wrapper_call.writeline(
            'RECORD_FUNCTION("inductor_wrapper_call", c10::ArrayRef<c10::IValue>({{}}));'
        )

    def make_buffer_allocation(self, buffer):
        from .cpp import DEVICE_TO_ATEN, DTYPE_TO_ATEN

        # TODO: map layout here
        device = buffer.get_device()
        dtype = buffer.get_dtype()
        shape = tuple(buffer.get_size())
        stride = tuple(buffer.get_stride())
        return (
            f"{self.declare}{buffer.get_name()} = {self.namespace}empty_strided("
            f"{self.codegen_shape_tuple(shape)}, "
            f"{self.codegen_shape_tuple(stride)}, "
            f"at::device({DEVICE_TO_ATEN[device.type]})"
            f".dtype({DTYPE_TO_ATEN[dtype]})){self.ending}"
        )

    def add_benchmark_harness(self, output):
        return


class CudaWrapperCodeGen(CppWrapperCodeGen):
    """
    Generates cpp wrapper for running on GPU and calls CUDA kernels
    """

    def __init__(self):
        super().__init__()
        self.kernel_callsite_id = 0
        self.arg_var_id = 0

    def write_prefix(self):
        self.prefix.splice(
            """
            #include <ATen/ATen.h>
            #include <ATen/cuda/CUDAContext.h>
            #include <ATen/cuda/Exceptions.h>
            #include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
            #include <c10/cuda/CUDAGuard.h>

            CUfunction loadKernel(const std::string &filePath, const std::string &funcName) {
                CUmodule mod;
                CUfunction func;
                AT_CUDA_DRIVER_CHECK(cuModuleLoad(&mod, filePath.c_str()));
                AT_CUDA_DRIVER_CHECK(cuModuleGetFunction(&func, mod, funcName.c_str()));
                return func;
            }

            void launchKernel(
                    CUfunction func,
                    int gridX,
                    int gridY,
                    int gridZ,
                    int numWraps,
                    int sharedMemBytes,
                    void* args[],
                    int device_index) {
                AT_CUDA_DRIVER_CHECK(cuLaunchKernel(
                    func, gridX, gridY, gridZ, 32*numWraps, 1, 1, sharedMemBytes,
                    at::cuda::getCurrentCUDAStream(device_index), args, nullptr));
            }
            """
        )

    def define_kernel(self, name: str, kernel: str, kernel_path: str = None):
        pass

    def generate(self):
        self.prefix.writeline("\n")
        for kernel in self.kernels.values():
            self.prefix.writeline(f"static CUfunction {kernel} = nullptr;")
        self.prefix.writeline("\n")
        return super().generate()

    def generate_load_kernel(self, name: str = None):
        params = KernelParamCache.cache.get(name, None)
        assert (
            params is not None
        ), "cuda kernel parameters should already exist at this moment"
        mangled_name = params.get("mangled_name", None)
        assert mangled_name is not None, "missing mangled_name"
        cubin_path = os.path.join(cubin_cache_dir(), f"{name}.cubin")
        assert os.path.exists(
            cubin_path
        ), "cubin file should already exist at this moment"

        self.writeline(f"if ({name} == nullptr) {{")
        self.writeline(
            f"""     {name} = loadKernel("{cubin_path}", "{mangled_name}");"""
        )
        self.writeline("}")

    def generate_args_decl(self, call_args):
        # TODO: only works for constant now, need type info
        new_args = []
        for arg in call_args:
            var_name = f"var_{self.arg_var_id}"
            if is_int(arg):
                self.writeline(f"int {var_name} = {arg};")
            elif is_float(arg):
                self.writeline(f"float {var_name} = {arg};")
            else:
                self.writeline(
                    f"CUdeviceptr {var_name} = reinterpret_cast<CUdeviceptr>({arg}.data_ptr());"
                )
            new_args.append(f"&{var_name}")
            self.arg_var_id += 1

        return ", ".join(new_args)

    def generate_kernel_call(self, name, call_args, device_index):
        params = KernelParamCache.cache.get(name, None)
        assert (
            params is not None
        ), "cuda kernel parameters should already exist at this moment"
        grid_x = params.get("grid_x", None)
        grid_y = params.get("grid_y", None)
        grid_z = params.get("grid_z", None)
        num_warps = params.get("num_warps", None)
        shared_mem = params.get("shared_mem", None)

        self.generate_load_kernel(name)

        call_args = self.generate_args_decl(call_args)
        args_name = f"kernel_args_{self.kernel_callsite_id}"
        self.kernel_callsite_id += 1
        self.writeline(f"void* {args_name}[] = {{{call_args}}};")
        self.writeline(
            f"launchKernel({name}, {grid_x}, {grid_y}, {grid_z}, {num_warps}, {shared_mem}, {args_name}, {device_index});"
        )
