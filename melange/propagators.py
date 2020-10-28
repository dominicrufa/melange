"""
dynamics propagators
"""
from jax import grad, vmap, jit, random
import jax.numpy as jnp
from melange.gaussians import *

def EL_mu_sigma(x, potential, dt, parameters):
    """
    create mean vector and covariance matrix for multivariate gaussian proposal
    mu = x - 0.5*dt*grad(potential(x | parameters))
    cov = dt*I

    arguments:
        x : jnp.array(N)
            current position (or latent variable)
        potential : function
            potential function (takes args x and parameters)
        dt : float
            incremental time
        parameters : jnp.array
            parameters passed to potential

    returns
        mu : jnp.array(N)
            mean of output
        Sigma : jnp.array(N,N)
            covariance matrix
    """
    tau = dt/2.
    force = -grad(potential)(x, parameters)
    mu = x + tau * force
    Sigma = 2*tau*jnp.eye(len(x))
    return mu, Sigma

def ULA_move(x, potential, dt, key, potential_parameter):
    """
    unadjusted langevin algorithm move

    arguments:
        x : jnp.array(N)
            current position (or latent variable)
        potential : function
            potential function (takes args x and parameters)
        dt : float
            incremental time
        key : float
            randomization key
        potential_parameters : jnp.array
            parameters passed to potential

    returns
        out : jnp.array(N)
            multivariate gaussian proposal
    """
    mu, cov = EL_mu_sigma(x, potential, dt, potential_parameter)
    return random.multivariate_normal(key, mu, cov)

def driven_Langevin_parameters(x, potential, dt, A_function, b_function, potential_parameter, A_parameter, b_parameter):
    """
     helper function to compute compute the mean and covariance matrix of a driven langevin move

     arguments:
        x : jnp.array(N)
            current position (or latent variable)
        potential : function
            potential function (takes args x and parameters)
        dt : float
            incremental time
        A_function : function
            covariance driver function
        b_function : function
            mean driver function
        potential_parameter : jnp.array
            parameters passed to potential
        A_parameter : jnp.array()
            second argument of A function
        b_parameter : jnp.array()
            second argument of b function
    returns
        mu : jnp.array(N)
            mean vector
        cov : jnp.array(N,N)
            covariance matrix
    """
    dimension = x.shape[0]
    #compute f_t
    tau = dt/2.
    force = -grad(potential)(x, potential_parameter)
    f_t = x + tau * force

    #compute theta_t
    _A = A_function(x, A_parameter)
    theta_t = jnp.linalg.inv(jnp.eye(dimension) + 2*dt * _A) #must be positive definite
    #compute mu
    # mu = jnp.matmul(theta_t, f_t - dt*b_function(x, b_parameter))
    # #compute cov
    # cov = dt*theta_t
    return _A, b_function(x, b_parameter), f_t, theta_t


def driven_Langevin_move(x, potential, dt, A_function, b_function, potential_parameter, A_parameter, b_parameter, key):
    """
    driven Langevin Algorithm; a driven langevin propagator is
    N(x_t; \Theta_t^i(x_{t-1})*(f_t-dt*b_t^i)(x_{t-1}), dt*\Theta_t^i(x_{t-1}))
    where \Theta_t^i(x_{t-1}) = (I_d + 2*dt*A_t^i(x_{t-1}))^{-1},
    f_t(x_{t-1}) = x_{t-1} + 0.5*dt*\nabla \pi_t(x_{t-1})

    arguments:
        x : jnp.array(N)
            current position (or latent variable)
        potential : function
            potential function (takes args x and parameters)
        dt : float
            incremental time
        A_function : function
            covariance driver function
        b_function : function
            mean driver function
        potential_parameter : jnp.array
            parameters passed to potential
        A_parameter : jnp.array()
            second argument of A function
        b_parameter : jnp.array()
            second argument of b function
        key : float
            randomization key

    returns
        out : jnp.array(N)
            multivariate gaussian proposal
    """
    A, b, f, theta = driven_Langevin_parameters(x, potential, dt, A_function, b_function, potential_parameter, A_parameter, b_parameter)
    mu, cov = driven_mu_cov(b, f, theta, dt)
    return random.multivariate_normal(key, mu, cov)

def log_Euler_Maruyma_kernel(x_tm1, x_t, potential, potential_parameters, dt):
    """
    the log kernel probability of the transition
    arguments
        x_tm1 : jnp.array(N)
            previous iteration position (of dimension N)
        x_t : jnp.array(N)
            current iteration positions (of dimension N)
        potential : function
            potential function
        potential_parameters : jnp.array(Q)
            second argument to potential function
        dt : float
            time increment

    returns
        logp : float
            the log probability of the kernel
    """
    mu, cov = EL_mu_sigma(x_tm1, potential, dt, potential_parameters)
    logp = multivariate_gaussian_logp(x_t, mu, cov)
    return logp

def log_driven_Langevin_kernel(x_tm1, x_t, potential, dt, A_function, b_function, potential_parameter, A_parameter, b_parameter):
    """
    compute the log of the driven langevin kernel transition probability
    arguments
        x_tm1 : jnp.array(N)
            previous iteration position (of dimension N)
        x_t : jnp.array(N)
            current iteration positions (of dimension N)
        potential : function
            potential function
        dt : float
            time increment
        A_function : function
            variance_controlling function
        b_function : function
            mean_controlling function
        potential_parameter : jnp.array(Q)
            second argument to potential function
        A_parameter :  jnp.array(R)
            argument to A_function
        b_parameter : jnp.array(S)
            argument to b_function

    return
        logp : float
            log probability
    """
    A, b, f, theta = driven_Langevin_parameters(x_tm1, potential, dt, A_function, b_function, potential_parameter, A_parameter, b_parameter)
    mu, cov = driven_mu_cov(b, f, theta, dt)
    logp = multivariate_gaussian_logp(x_t, mu, cov)
    return logp


def Euler_Maruyama_log_proposal_ratio(x_tm1,
                                      x_t,
                                      forward_potential,
                                      forward_potential_parameters,
                                      forward_dt,
                                      backward_potential,
                                      backward_potential_parameters,
                                      backward_dt):
    """
    compute log(L_{t-1}(x_t, x_{t-1})) - log(K_t(x_{t-1}, x_t))
    for EL kernels

    arguments
        x_tm1 : jnp.array(N)
            previous iteration position (of dimension N)
        x_t : jnp.array(N)
            current iteration positions (of dimension N)
        forward_potential : function
            forward kernel potential function
        forward_potential_parameters : jnp.array(Q)
            forward kernel parameter
        forward_dt : float
            forward kernel time increment
        backward_potential : function
            backward kernel potential function
        backward_potential_parameters : jnp.array(Q)
            backward kernel parameter
        backward_dt : float
            backward kernel time increment

    returns
        out : float
            log ratio of the backward-to-forward proposal
    """
    logK = log_Euler_Maruyma_kernel(x_tm1, x_t, forward_potential, forward_potential_parameters, forward_dt)
    logL = log_Euler_Maruyma_kernel(x_t, x_tm1, backward_potential, backward_potential_parameters, backward_dt)
    return logL - logK

def driven_Langevin_log_proposal_ratio(x_tm1,
                                      x_t,
                                      forward_potential,
                                      backward_potential,
                                      forward_dt,
                                      backward_dt,
                                      A_function,
                                      b_function,
                                      forward_potential_parameter,
                                      backward_potential_parameter,
                                      A_parameter,
                                      b_parameter):
    """
    arguments
        x_tm1 : jnp.array(N)
            previous iteration position (of dimension N)
        x_t : jnp.array(N)
            current iteration positions (of dimension N)
        forward_potential : function
            forward kernel potential function
        backward_potential : function
            backward kernel potential function
        forward_dt : float
            forward kernel time increment
        backward_dt : float
            backward kernel time increment
        A_function : function
            variance_controlling function
        b_function : function
            mean_controlling function
        forward_potential_parameter : jnp.array(Q)
            second argument to potential function
        backward_potential_parameter : jnp.array(R)
            backward kernel parameter
        A_parameter :  jnp.array(R)
            argument to A_function
        b_parameter : jnp.array(S)
            argument to b_function

    returns
        out : float
            log ratio of the backward-to-forward proposal


    """
    logK = log_driven_Langevin_kernel(x_tm1, x_t, forward_potential, forward_dt, A_function, b_function, forward_potential_parameter, A_parameter, b_parameter)
    logL = log_Euler_Maruyma_kernel(x_t, x_tm1, backward_potential, backward_potential_parameter, backward_dt)
    return logL - logK

def forward_ULA_sampler(xs, potential, dts, key, potential_parameters):
    """
    conduct forward sampling

    arguments
        xs : jnp.array(M,N)
            positions (or latent variables) of starting positions (M particles with dimension N)
        potential : function
            potential function (takes args x and parameters)
        dts : float or jnp.array
            incremental time; if jnp.array, must be same dimension as `potential_parameters`
        key : float
            randomization key
        potential_parameters : jnp.array
            parameters passed to potential

    return
        trajectories : jnp.array(T,N,M)
            trajectories of N particles (each with dimension M) at T steps of integration
    """
    from jax import vmap
    from jax.lax import scan
    num_particles, dimension = xs.shape
    sequence_length = len(potential_parameters)

    if type(dts) == float:
        dts = jnp.array([dts]*sequence_length)
    assert len(dts) == sequence_length

    # args for ULA_move are : (x, potential, dt, key, potential_parameter)
    vmap_ULA_move = vmap(ULA_move, in_axes=(0, None, None, 0, None))

    def ULA_scan(in_xs_and_key, t):
        in_xs, key = in_xs_and_key
        run_dt, propagation_parameter = dts[t], potential_parameters[t]
        key_folder = random.split(key, num_particles+1)
        out_key, run_key = key_folder[0], key_folder[1:]
        # args for ULA_move are : (x, potential, dt, key, potential_parameter)
        assert len(run_key) == len(in_xs)
        out_xs = vmap_ULA_move(in_xs, potential, run_dt, run_key, propagation_parameter)
        return (out_xs, out_key), out_xs
    key, init_run_key = random.split(key)
    init = (xs, init_run_key)
    _, trajectories = scan(ULA_scan, init, jnp.arange(sequence_length))
    return trajectories

def forward_driven_diffusion_sampler(xs,
                                     potential,
                                     dt,
                                     key,
                                     A_function,
                                     b_function,
                                     potential_parameters,
                                     A_parameters,
                                     b_parameters):

    """
    arguments
        xs : jnp.array(M,N)
            positions (or latent variables) of starting positions (M particles with dimension N)
        potential : function
            potential function (takes args x and parameters)
        dt : float
            incremental time
        key : float
            randomization key
        A_function : function
            covariance driver function
        b_function : function
            mean driver function
        potential_parameters : jnp.array
            parameters passed to potential
        A_parameters : jnp.array()
            second argument of A function
        b_parameters : jnp.array()
            second argument of b function
    """
    from jax import vmap
    from jax.lax import scan

    num_particles, dimension = xs.shape
    assert len(potential_parameters) == len(A_parameters)
    assert len(potential_parameters) == len(b_parameters)
    vmap_driven_move = vmap(driven_Langevin_move,
                          in_axes=(0, None, None, None, None, None, None, None, 0))
    sequence_counter = jnp.arange(0,len(potential_parameters))
    root_keys = random.split(key, len(potential_parameters))
    def driven_scan(in_xs, iteration):
        run_keys = random.split(root_keys[iteration], num_particles)
        potential_parameter = potential_parameters[iteration]
        A_parameter, b_parameter = A_parameters[iteration] , b_parameters[iteration]
        out_xs = vmap_driven_move(in_xs, potential, dt, A_function, b_function, potential_parameter, A_parameter, b_parameter, run_keys)
        return out_xs, out_xs

    _, trajectories = scan(driven_scan, xs, sequence_counter)
    return trajectories

def generate_Euler_Maruyama_propagators():
    """
    importer function
    function that creates two functions:
    1. first function created is a kernel propagator (K)
    2. second function returns the kernel ratio calculator
    """
    # let's make the kernel propagator first: this is just a batched ULA move
    kernel_propagator = ULA_move

    # let's make the kernel ratio calculator
    kernel_ratio_calculator = Euler_Maruyama_log_proposal_ratio

    #return both
    return kernel_propagator, kernel_ratio_calculator

def generate_driven_Langevin_propagators():
    """
    importer function
    create 2 functions...
    1. forward kernel propagator
    2. kernel ratio calculator for driven langevin dynamics
    """
    kernel_propagator = driven_Langevin_move
    kernel_ratio_calculator = driven_Langevin_log_proposal_ratio

    return kernel_propagator, kernel_ratio_calculator

def driven_mu_cov(b, f, theta, dt):
    """
    compute the twisted mean and covariance matrix of twisting parameters

    arguments
        b : jnp.array(N)
            twisting vector
        f : jnp.array(N)
            push vector
        theta : jnp.array(N,N)
            twist matrix
        dt : float
            time increment
    returns
        mu : jnp.array(N)
            twisted mean
        cov : jnp.array(N,N)
            twisted covariance matrix
    """
    mu = jnp.matmul(theta, f - dt*b)
    cov = dt*theta
    return mu, cov
