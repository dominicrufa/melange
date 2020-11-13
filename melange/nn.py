"""
some neural network utilities that i lifted from jax
"""
from jax import numpy as jnp
import jax
from jax.config import config; config.update("jax_enable_x64", True)

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
    if key==None:
        return jnp.zeros((n,m)), jnp.zeros(n)
    else:
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key, scale=1e-2):
    """
    a simple wrapper for `random_layer_params` that will initialize tuples of weights and biases for a fully connected neural network;
    the first and last entries in the sizes list should be the input and output dimensions of the neural network

    arguments
        sizes : list of int
            list of layer sizes for fully connected neural network
        key : random.PRNGKey, default None
            key to randomize layer parameters;
            if None, all are set to 0
        scale : float
            factor by which to scale random layer parameters

    returns
        out : list of tuples of arrays
            initialized parameters
    """
    if key is None:
        keys=[None]*len(sizes)
    else:
        keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k, scale) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def arrayize_nn_params(nn_params):
    """
    turn a list of tuples of (W,b) into a 1D array
    """
    shapes = [(w.shape, b.shape) for w,b in nn_params]
    combined_wbs = [jnp.append(w, b[..., jnp.newaxis], axis=1) for w,b in nn_params]
    outs = jnp.concatenate(([a.flatten() for a in combined_wbs]))
    return outs

def simple_nnpotential_generator(base_potential, nn_potential, base_potential_parameter_length, nn_potential_param_list):
    """
    nn potential function generator that allows for the passage of a jnp.array(R) as the second argument for a fully-connected neural network

    Example:
    >>> nn_params = init_network_params([1,4,4], None) # make starting params template of zeros
    >>> nn_param_array = arrayize_nn_params(nn_params) # make the nn params a 1D array
    >>> combined_potential = simple_nnpotential_generator(potential, sigmoid_nn_potential, 1, nn_params) # combine the default potential with the sigmoid potential
    >>>  combined_potential(jnp.array([2.]), np.concatenate((jnp.array([0.]), nn_param_array))) #compute the energy at 2.

    arguments
        base_potential : function
            potential that returns the non-nn-perturbed energy
        nn_potential : function
            potential that returns an nn-energy
        base_potential_parameter_length : int
            length of arg 1 to the base potential
        nn_potential_param_list : list of tup of jnp.array()
            a list of tuples corresponding to (W,b) for a fully-connected neural network parameterization

    returns
        out_nnpotential : function
            combined function that takes a 1d array as arg1
    """

    shapes = [(w.shape, b.shape) for w,b in nn_potential_param_list]
    def out_nnpotential(x, parameter):
        base_e_param, nn_param = parameter[:base_potential_parameter_length], parameter[base_potential_parameter_length:]

        #resolve the base param
        base_e = base_potential(x, base_e_param)

        #reshape the nn_param
        nn_param_list = []
        starter = 0
        for entry in shapes:
            Wrows, Wcols = entry[0]
            comb = nn_param[starter: starter + Wrows*Wcols + Wrows]
            comb = comb.reshape(Wrows, Wcols + 1)
            W, b = comb[:,:-1], comb[:,-1].flatten()
            nn_param_list.append((W,b))
            starter = starter + Wrows*Wcols + Wrows

        #call the nn_potential
        nn_e = nn_potential(x, nn_param_list)
        return nn_e + base_e

    return out_nnpotential

def sigmoid_nn_potential(pos, parameter):
    """
    make a simple, fully-connected
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

    #WARNING : argument-value independent since the potential returns a scalar. you CANNOT allocate dynamically-sized parameters here
    """
    act = pos
    for i in range(len(parameter)):
        w,b = parameter[i]
        dotter = jnp.dot(w, act)
        outs = dotter + b
        act = jax.nn.sigmoid(outs) - 0.5
    return act.sum()
