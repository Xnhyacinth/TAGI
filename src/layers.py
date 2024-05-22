from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union
from typing_extensions import TypeAlias
import jax.numpy as jnp
from jax import lax
from flax.linen import partitioning as nn_partitioning
from flax import linen
import functools
import operator
import numpy as np
# Type annotations
Array: TypeAlias = jnp.ndarray
DType: TypeAlias = jnp.dtype
PRNGKey: TypeAlias = jnp.ndarray
Shape = Iterable[int]
Activation = Callable[..., Array]
# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]


def hypernetwork(output, name, emb_dim, activations=('relu',)):
    # if cfg.per_layer_hnet:
    #     output *= cfg.num_encoder_layers + 2 * cfg.num_decoder_layers
    return MlpBlock(
        intermediate_dim=emb_dim,  # same size as model
        output_dim=output,
        activations=activations,
        dtype='bfloat16',
        name=name,
    )

# from flax.linen.partitioning import param_with_axes, with_sharding_constraint
param_with_axes = nn_partitioning.param_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
    # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
    return tuple([ax if ax >= 0 else ndim + ax for ax in axes])


def _canonicalize_tuple(x):
    if isinstance(x, Iterable):
        return tuple(x)
    else:
        return (x,)


def _convert_to_activation_function(
    fn_or_string: Union[str, Callable]
) -> Callable:
    """Convert a string to an activation function."""
    if fn_or_string == 'linear':
        return lambda x: x
    elif isinstance(fn_or_string, str):
        return getattr(linen, fn_or_string)
    elif callable(fn_or_string):
        return fn_or_string
    else:
        raise ValueError(
            "don't know how to convert %s to an activation function"
            % (fn_or_string,)
        )

# ------------------------------------------------------------------------------
# DenseGeneral for attention layers.
# ------------------------------------------------------------------------------

class DenseGeneral(linen.Module):
    """A linear transformation (without bias) with flexible axes.

    Attributes:
      features: tuple with numbers of output features.
      axis: tuple with axes to apply the transformation on.
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer function for the weight matrix.
    """

    features: Union[Iterable[int], int]
    axis: Union[Iterable[int], int] = -1
    dtype: DType = jnp.float32
    kernel_init: Initializer = linen.initializers.variance_scaling(
        1.0, 'fan_in', 'truncated_normal'
    )
    kernel_axes: Tuple[str, ...] = ()

    @linen.compact
    def __call__(self, inputs: Array) -> Array:
        """Applies a linear transformation to the inputs along multiple dimensions.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        features = _canonicalize_tuple(self.features)
        axis = _canonicalize_tuple(self.axis)

        inputs = jnp.asarray(inputs, self.dtype)
        axis = _normalize_axes(axis, inputs.ndim)

        kernel_shape = tuple([inputs.shape[ax] for ax in axis]) + features
        kernel_param_shape = (
            np.prod([inputs.shape[ax] for ax in axis]),
            np.prod(features),
        )
        kernel = param_with_axes(
            'kernel',
            self.kernel_init,
            kernel_param_shape,
            jnp.float32,
            axes=self.kernel_axes,
        )
        kernel = jnp.asarray(kernel, self.dtype)
        kernel = jnp.reshape(kernel, kernel_shape)

        contract_ind = tuple(range(0, len(axis)))
        return lax.dot_general(inputs, kernel, ((axis, contract_ind), ((), ())))


class MlpBlock(linen.Module):
    """Transformer MLP / feed-forward block.

    Attributes:
      intermediate_dim: Shared dimension of hidden layers.
      activations: Type of activations for each layer.  Each element is either
        'linear', a string function name in flax.linen, or a function.
      kernel_init: Kernel function, passed to the dense layers.
      deterministic: Whether the dropout layers should be deterministic.
      intermediate_dropout_rate: Dropout rate used after the intermediate layers.
      dtype: Type for the dense layer.
    """
    intermediate_dim: int = 2048
    output_dim: Optional[int] = None  # by default we preserve the input dim
    activations: Sequence[Union[str, Callable]] = ("relu",)
    kernel_init: Initializer = linen.initializers.variance_scaling(
        1.0, "fan_in", "truncated_normal")
    intermediate_dropout_rate: float = 0.1
    dtype: Any = jnp.float32

    @linen.compact
    def __call__(self, inputs, decode: bool = False, deterministic: bool = False):
        """Applies Transformer MlpBlock module."""
        # Iterate over specified MLP input activation functions.
        # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
        activations = []
        for idx, act_fn in enumerate(self.activations):
            dense_name = "wi" if len(self.activations) == 1 else f"wi_{idx}"
            x = DenseGeneral(
                self.intermediate_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                kernel_axes=("embed", "mlp"),
                name=dense_name,
            )(inputs)
            x = _convert_to_activation_function(act_fn)(x)
            activations.append(x)

        # Take elementwise product of above intermediate activations.
        x = functools.reduce(operator.mul, activations)
        # Apply dropout and final dense output projection.
        x = linen.Dropout(rate=self.intermediate_dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic
        )  # Broadcast along length.

        # CHANGE from t5x
        # Removing the sharding constraint as we require to use this layer for shapes of ('batch', 'mlp'),
        # which makes below constraint invalid.
        # x = with_sharding_constraint(x, ('batch', 'length', 'mlp'))

        output = DenseGeneral(
            inputs.shape[-1] if self.output_dim is None else self.output_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_axes=("mlp", "embed"),
            name="wo",
        )(x)
        return output