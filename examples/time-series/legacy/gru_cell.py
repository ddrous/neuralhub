
########### GRU CELL, MOVE THIS TO A SEPARATE FILE
def default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    else:
        return jnp.float32
def default_init(
    key, shape, dtype, lim
) -> jax.Array:
    if jnp.issubdtype(dtype, jnp.complexfloating):
        real_dtype = jnp.finfo(dtype).dtype
        rkey, ikey = jrandom.split(key, 2)
        real = jrandom.uniform(rkey, shape, real_dtype, minval=-lim, maxval=lim)
        imag = jrandom.uniform(ikey, shape, real_dtype, minval=-lim, maxval=lim)
        return real.astype(dtype) + 1j * imag.astype(dtype)
    else:
        return jrandom.uniform(key, shape, dtype, minval=-lim, maxval=lim)

Array = jnp.ndarray
import jax.nn as jnn
class MyGRUCell(eqx.Module, strict=True):
    """A special single step of a Gated Recurrent Unit (GRU).
    """
    weight_ih: Array
    weight_hh: Array
    bias: Array
    bias_n: Array
    input_size: int 
    hidden_size: int
    use_bias: bool 

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True,
        dtype=None,
        *,
        key: jax.random.PRNGKey,
    ):
        """**Arguments:**
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        ihkey, hhkey, bkey, bkey2 = jrandom.split(key, 4)
        lim = math.sqrt(1 / hidden_size)

        fn = eqx.nn.de
        ihshape = (3 * hidden_size, input_size)
        self.weight_ih = default_init(ihkey, ihshape, dtype, lim)
        hhshape = (3 * hidden_size, hidden_size)
        self.weight_hh = default_init(hhkey, hhshape, dtype, lim)
        if use_bias:
            self.bias = default_init(bkey, (3 * hidden_size,), dtype, lim)
            self.bias_n = default_init(bkey2, (hidden_size,), dtype, lim)
        else:
            self.bias = None
            self.bias_n = None

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias

    def __call__(
        self, input, hidden, *, key = None
    ):
        """**Arguments:**
        The updated hidden state, which is a JAX array of shape `(hidden_size,)`.
        """
        if self.use_bias:
            bias = self.bias
            bias_n = self.bias_n
        else:
            bias = 0
            bias_n = 0
        igates = jnp.split(self.weight_ih @ input + bias, 3)
        hgates = jnp.split(self.weight_hh @ hidden, 3)
        reset = jnn.sigmoid(igates[0] + hgates[0])
        inp = jnn.sigmoid(igates[1] + hgates[1])
        new = jnn.tanh(igates[2] + reset * (hgates[2] + bias_n))
        return new + inp * (hidden - new)

##############################################

