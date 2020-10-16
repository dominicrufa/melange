"""
test melange.propagators
"""
from jax import random
from jax import vmap
import jax.numpy as jnp
from melange.propagators import *
from melange.tests.utils import checker_function, get_nondefault_potential_initializer
import tqdm
import numpy as np

def test_1D_ULA_propagator(key = random.PRNGKey(0), num_runs=1000):
    """
    take a batch of 1000 particles distributed according to N(0,2), run dynamics with ULA for 1000 steps with dt=0.01 on a potential whose invariant is N(0,2)
    and assert that the mean and variance is unchanged within a tolerance
    """
    key, genkey = random.split(key)
    potential, (mu, cov), dG = get_nondefault_potential_initializer(1)
    x_ula_starter = random.multivariate_normal(key = genkey, mean = mu, cov = cov, shape=[num_runs])
    dt=1e-2
    batch_ula_move = vmap(ULA_move, in_axes=(0, None, None, 0, None))
    potential_parameter = jnp.array([0.])

    for i in tqdm.trange(100):
        key, ula_keygen = random.split(key, 2)
        ula_keys = random.split(ula_keygen, num_runs)
        x_ULA = batch_ula_move(x_ula_starter, potential, dt, ula_keys, potential_parameter)
        x_ula_starter = x_ULA

    ula_mean, ula_std = x_ula_starter.mean(), x_ula_starter.std()

    assert checker_function(ula_mean,0.2)
    assert checker_function(ula_std - jnp.sqrt(2), 0.2)

def test_1D_driven_propagator(key = random.PRNGKey(0), num_runs=1000):
    """
    take a batch of 1000 particles distributed according to N(0,2), run dynamics with driven langevin algorithm for 1000 steps with dt=0.01 on a potential whose invariant is N(0,2)
    and assert that the mean and variance is unchanged within a tolerance.
    """
    key, genkey = random.split(key)
    potential, (mu, cov), dG = get_nondefault_potential_initializer(1)
    x_driven_starter = random.multivariate_normal(key = genkey, mean = mu, cov = cov, shape=[num_runs])
    dt=1e-2
    #make dummy A and b functions
    def A(x, a_param): return jnp.zeros((x.shape[0], x.shape[0]))
    def b(x, b_param): return jnp.zeros(x.shape[0])
    batch_driver_move = vmap(driven_Langevin_move, in_axes=(0,None,None,None,None,None,None,None,0))
    potential_parameter = jnp.array([0.])

    for i in tqdm.trange(100):
        key, drive_keygen = random.split(key, 2)
        drive_keys = random.split(drive_keygen, num_runs)
        x_drive = batch_driver_move(x_driven_starter,
                                  potential,
                                  dt,
                                  A,
                                  b,
                                  potential_parameter,
                                  jnp.array([0.]),
                                  jnp.array([0.]),
                                  drive_keys)
        x_driven_starter = x_drive

    driven_mean, driven_std = x_driven_starter.mean(), x_driven_starter.std()

    assert checker_function(driven_mean,0.2)
    assert checker_function(driven_std - jnp.sqrt(2), 0.2)

def test_1d_kernel_consistency(key = random.PRNGKey(0)):
    """
    with a 'dummy' driven forward kernel, assert that the log forward probability
    is equal to that of the ULA propagator in one dimension
    """
    from melange.propagators import generate_Euler_Maruyama_propagators, generate_driven_Langevin_propagators, Euler_Maruyama_log_proposal_ratio, driven_Langevin_log_proposal_ratio

    dt=0.1
    forward_potential_parameters= jnp.array([0.])
    backward_potential_parameters = jnp.array([0.])

    #make dummy A and b functions
    def A(x, a_param): return jnp.zeros((x.shape[0], x.shape[0]))
    def b(x, b_param): return jnp.zeros(x.shape[0])

    potential, (mu, cov), dG = get_nondefault_potential_initializer(1)
    xs = random.multivariate_normal(key = key, mean = jnp.array([1.]), cov = jnp.array([[1.]]), shape=[2])

    EM_propagator, EM_kernel = generate_Euler_Maruyama_propagators()
    driven_propagator, driven_kernel = generate_driven_Langevin_propagators()

    EM_logp_ratio = Euler_Maruyama_log_proposal_ratio(xs[0], xs[1], potential, forward_potential_parameters, dt, potential, backward_potential_parameters, dt)
    driven_logp_ratio = driven_Langevin_log_proposal_ratio(xs[0],
                                      xs[1],
                                      potential,
                                      potential,
                                      dt,
                                      dt,
                                      A,
                                      b,
                                      forward_potential_parameters,
                                      backward_potential_parameters,
                                      A_parameter = forward_potential_parameters,
                                      b_parameter = forward_potential_parameters)
    assert np.isclose(EM_logp_ratio, driven_logp_ratio)

def test_forward_ULA_driven_samplers(key = random.PRNGKey(0)):
  """
  given a randomization key, execute `forward_ULA_sampler` and `forward_driven_diffusion_sampler`
  with a time-independent potential that has the same mean and variance as the distribution of (5000) initial
  samples. We only assert that the statistics of the post-propagated samples obey the same statistics (within a tolerance).
  """
  from melange.propagators import forward_ULA_sampler, forward_driven_diffusion_sampler

  dt=0.1
  potential_parameters= jnp.zeros((100,1))
  A_parameters = potential_parameters
  b_parameters = potential_parameters

  #make dummy A and b functions
  def A(x, a_param): return jnp.zeros((x.shape[0], x.shape[0]))
  def b(x, b_param): return jnp.zeros(x.shape[0])

  potential, (mu, cov), dG = get_nondefault_potential_initializer(1)
  xs = random.multivariate_normal(key = key, mean = mu, cov = cov, shape=[5000])
  og_mean, og_variance = xs.mean(), xs.var()
  #print(og_mean, og_variance)

  ULA_trajs = forward_ULA_sampler(xs, potential, dt, key, potential_parameters)
  #print(ULA_trajs[-1].mean(), ULA_trajs[-1].var())

  driven_trajs = forward_driven_diffusion_sampler(xs, potential, dt, key, A, b, potential_parameters, A_parameters, b_parameters)
  #print(driven_trajs[-1].mean(), driven_trajs[-1].var())

  mean_tolerance = 0.2
  assert checker_function(ULA_trajs[-1].mean(), mean_tolerance)
  assert checker_function(driven_trajs[-1].mean(), mean_tolerance)

  variance_tolerance = 0.2
  assert checker_function(ULA_trajs[-1].var() - 2., variance_tolerance)
  assert checker_function(driven_trajs[-1].var()-2., variance_tolerance)
