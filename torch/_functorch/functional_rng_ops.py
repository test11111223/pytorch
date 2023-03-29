import torch
from torch import _prims
from torch.utils._python_dispatch import TorchDispatchMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
)
from torch._subclasses.fake_tensor import disable_fake_tensor_mode_tracing
from typing import Tuple
from torch.types import _device, _dtype
from torch.fx.operator_schemas import normalize_function


aten = torch.ops.aten


def get_default_stride(size):
    """
    A helper function to get the strides for a contiguous tensor of a given
    shape.
    """
    stride = [1] * len(size) + [1]
    for idx in reversed(range(len(size))):
        stride[idx] = stride[idx + 1] * size[idx]
    stride = stride[1:]
    return stride


# New RNG ops
def _philox_rand(
    shape: torch.Size,
    seed: torch.Tensor,
    offset: torch.Tensor,
    stride: Tuple[int, ...],
    device: _device,
    dtype: _dtype,
):
    # FIXME - Need to add a nondeterministic_seeded tag to this op. Not sure how to do that yet.
    stride = tuple(stride)
    with torch.random.fork_rng(
        devices=[
            device,
        ]
    ):
        torch.manual_seed(seed)
        full_size = list(shape) + [stride[-1]]
        full_stride = stride + (1,)
        for i in reversed(range(len(full_stride))):
            if i == 0:
                full_size[i] = shape[0]
            else:
                assert full_stride[i - 1] % full_stride[i] == 0
                full_size[i] = full_stride[i - 1] // full_stride[i]

        for i in range(len(full_stride)):
            if offset % full_stride[i] == 0:
                full_size[i] += offset // full_stride[i]
                break
        else:
            assert False

        return torch.rand(full_size, device=device, dtype=dtype).as_strided(
            shape, stride, offset
        )


def _philox_rand_meta(
    shape: torch.Size,
    seed: torch.Tensor,
    offset: torch.Tensor,
    stride: Tuple[int, ...],
    device: _device,
    dtype: _dtype,
):
    # TODO - Update the state here
    return _prims.TensorMeta(shape=shape, strides=stride, dtype=dtype, device=device)


philox_rand = _prims._make_prim(
    schema="philox_rand(int[] size, Tensor seed, Tensor offset, int[] stride, Device? device=None, ScalarType? dtype=None) -> Tensor",
    return_type=_prims.RETURN_TYPE.NEW,
    meta=_philox_rand_meta,
    impl_aten=_philox_rand,
    tags=(torch.Tag.nondeterministic_seeded,),
    doc="",
)

class PhiloxRandomState:
    # These are the running seed and offset. We check them if rng state has been
    # modified by the user and we need to adjust.
    current_seed = -1
    current_offset = -1
    # This offset is used in the extracted AOT graph
    relative_offset = 0
    # These args are used by FunctionalizeRngOpsMode
    seed_arg = None
    base_offset_arg = None

    accumulated_fwd_offset = 0
    accumulated_bwd_offset = 0
    accumulated_offset = 0
    fwd_seed_arg = None
    fwd_base_offset_arg = None
    bwd_seed_arg = None
    bwd_base_offset_arg = None



    @staticmethod
    def get_current_seed_offset_scalar_tensors():
        with disable_proxy_modes_tracing():
            with disable_fake_tensor_mode_tracing():
                rng_state = torch.cuda.get_rng_state()
                seed = rng_state[800:808].view(dtype=torch.int64)[0]
                offset = rng_state[808:].view(dtype=torch.int64)[0]
                return seed, offset

    @classmethod
    def advance_rng_state(cls, mode):
        # with disable_proxy_modes_tracing():
        #     with disable_fake_tensor_mode_tracing():
        if mode == "forward":
            accumulated_offset = cls.accumulated_fwd_offset
        else:
            assert mode == "backward"
            accumulated_offset = cls.accumulated_bwd_offset
        print(f"{mode}, total offset = {accumulated_offset}")
        rng_state = torch.cuda.get_rng_state()
        seed = rng_state[800:808].view(dtype=torch.int64)[0]
        offset = rng_state[808:].view(dtype=torch.int64)[0]
        new_offset = offset + accumulated_offset
        torch.cuda.set_rng_state(cls.create_rng_state_tensor(seed, new_offset))

    @classmethod
    def mark_beginning_of_forward(cls):
        cls.seed_arg = cls.fwd_seed_arg
        cls.base_offset_arg = cls.fwd_base_offset_arg

    @classmethod
    def get_current_args(cls):
        # with disable_proxy_modes_tracing():
        with disable_fake_tensor_mode_tracing():
            seed_portion = cls.seed_arg.reshape([1])
            offset_portion = cls.base_offset_arg.reshape([1])
            return torch.cat([seed_portion, offset_portion])
        # return PhiloxRandomState.create_rng_state_tensor(cls.seed_arg, cls.base_offset_arg)

    @classmethod
    def reset_current_args(cls, x):
        with disable_fake_tensor_mode_tracing():
            seed, offset = torch.split(x, 1)
            cls.seed_arg = seed[0]
            cls.base_offset_arg = offset[0]
            # seed_portion = cls.seed_arg.reshape([1]).view(torch.uint8)
            # offset_portion = cls.base_offset_arg.reshape([1]).view(torch.uint8)
            # return torch.cat([seed_portion, offset_portion])

    @classmethod
    def mark_beginning_of_backward(cls):
        cls.accumulated_fwd_offset = cls.accumulated_offset
        cls.accumulated_offset = 0
        cls.seed_arg = cls.bwd_seed_arg
        cls.base_offset_arg = cls.bwd_base_offset_arg

    @classmethod
    def mark_end_of_backward(cls):
        cls.accumulated_bwd_offset = cls.accumulated_offset

    @classmethod
    def record_rng_state_args(cls, seed, offset, mode):
        if mode == "forward":
            cls.fwd_seed_arg = seed
            cls.fwd_base_offset_arg = offset
        else:
            assert mode == "backward"
            cls.bwd_seed_arg = seed
            cls.bwd_base_offset_arg = offset

    @classmethod
    def reset(cls):
        cls.seed_arg = None
        cls.base_offset_arg = None
        cls.accumulated_offset = 0
        cls.accumulated_fwd_offset = 0
        cls.accumulated_bwd_offset = 0

    @staticmethod
    def get_offset_jump(shape):
        # TODO - Specific to PyTorch CUDA impl. It calculates the total number
        # of randoms generated by CUDA. If everything fits nicely in the
        # stride-loop CUDA kernel, this is equal to the number of elements. But,
        # when a thread block has some unusable threads, it can be a different
        # number.

        # For impl, look at calc_execution_policy
        numel = 1
        for dim_size in shape:
            numel *= dim_size

        # TODO - Some bug in the following code, so returning numel for now
        return numel

        block_size = 256
        unroll = 4
        curand4_engine_calls = 4
        device_property = torch.cuda.get_device_properties(torch.cuda.current_device())
        blocks_per_sm = int(
            device_property.max_threads_per_multi_processor / block_size
        )
        grid_size = int((numel + block_size - 1) / block_size)
        grid_size = min(
            grid_size, device_property.multi_processor_count * blocks_per_sm
        )
        offset = (
            int((numel - 1) / (block_size * grid_size * unroll) + 1)
            * curand4_engine_calls
        )
        print(numel, offset)
        return offset

    @staticmethod
    def create_rng_state_tensor(seed, offset):
        seed_portion = seed.reshape([1]).view(torch.uint8)
        offset_portion = offset.reshape([1]).view(torch.uint8)
        prefix = torch.tensor([-1] * 800, dtype=torch.uint8)
        return torch.cat([prefix, seed_portion, offset_portion])

    @classmethod
    def get_state_args(cls, shape):
        with disable_proxy_modes_tracing():
            with disable_fake_tensor_mode_tracing():
                old_offset = cls.accumulated_offset
                offset_jump = PhiloxRandomState.get_offset_jump(shape)
                cls.accumulated_offset += offset_jump
        return cls.seed_arg, torch.add(cls.base_offset_arg, old_offset)


class FunctionalizeRngOpsMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        _, new_kwargs = normalize_function(
            func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
        )
        if func in [aten.rand.default]:
            shape = args[0]
            seed, offset = PhiloxRandomState.get_state_args(shape)
            device = new_kwargs["device"] or "cpu"
            dtype = new_kwargs["dtype"] or torch.float32
            stride = get_default_stride(shape)
            r = philox_rand(shape, seed, offset, stride, device, dtype)
            return r
        elif func in [aten.rand_like.default]:
            x = args[0]
            seed, offset = PhiloxRandomState.get_state_args(x.shape)
            r = philox_rand(x.shape, seed, offset, x.stride(), x.device, x.dtype)
            return r
        return func(*args, **kwargs)
