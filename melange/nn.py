"""
some neural network utilities that i lifted from jax
"""
from jax import numpy as np
import jax

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
    if key==None:
        return jnp.zeros((n,m)), jnp.zeros(n)
    else:
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
    if key is None:
        keys=[None]*len(sizes)
    else:
        keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def simple_neural_potential(pos, parameter):
    """
    arguments
        pos : jnp.array(R)
            latent variable
        parameter : (param1, param2)
            param1 : float
            param2 : tup
                tuple of neural network params (W,b)
    returns
        out : float
            potential value
    """
    mod_parameter = parameter[0]
    params = parameter[1]
    x0 = jnp.ones(pos.shape)*5 # at lambda=1, the mean is 3.
    base_potential = (1. - mod_parameter)*jnp.dot(pos, pos) + mod_parameter*jnp.dot(pos - x0, pos - x0)

    activations=pos
    for w, b in params:
        outputs = jnp.dot(w, activations) + b
        activations = jax.nn.sigmoid(outputs) - 0.5

    return base_potential + activations.sum()
