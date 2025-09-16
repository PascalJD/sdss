import jax.numpy as jnp
import jax

def get_linear_noise_schedule(total_steps, sigma_min=0.01, sigma_max=10., reverse=True):
    if reverse:
        def linear_noise_schedule(step):
            t = step / total_steps
            return (1 - t) * sigma_min + t * sigma_max
            # return 0.5 * ((1 - t) * sigma_min + t * sigma_max)
    else:
        def linear_noise_schedule(step):
            t = (total_steps - step) / total_steps
            return (1 - t) * sigma_min + t * sigma_max
            # return 0.5 * ((1 - t) * sigma_min + t * sigma_max)
    return linear_noise_schedule


def get_cosine_noise_schedule(total_steps, sigma_min=0.01, sigma_max=10., s=0.008, pow=2, reverse=True):
    if reverse:
        def cosine_noise_schedule(step):
            t = step / total_steps
            offset = 1 + s
            return 0.5 * (sigma_max - sigma_min) * jnp.cos(0.5 * jnp.pi * (offset - t) / offset) ** pow + 0.5 * sigma_min
    else:
        def cosine_noise_schedule(step):
            t = (total_steps - step) / total_steps
            offset = 1 + s
            return 0.5 * (sigma_max - sigma_min) * jnp.cos(0.5 * jnp.pi * (offset - t) / offset) ** pow + 0.5 * sigma_min
    return cosine_noise_schedule


def get_constant_noise_schedule(value, reverse=True):
    def constant_noise_schedule(step):
        return jnp.array(value)
    return constant_noise_schedule


def build_linear_sigmas(
    num_steps: int,
    sigma_min: float = 0.01,
    sigma_max: float = 10.0,
    reverse: bool = True,
) -> jnp.ndarray:
    i = jnp.arange(num_steps, dtype=jnp.float32)
    denom = jnp.maximum(num_steps - 1, 1)
    if reverse:
        t = i / denom
        sigmas = (1.0 - t) * sigma_max + t * sigma_min
    else:
        t = i / denom
        sigmas = (1.0 - t) * sigma_min + t * sigma_max
    sigmas = jnp.concatenate([sigmas, jnp.array([0.0], dtype=jnp.float32)], axis=0)
    return sigmas


def build_cosine_sigmas(
    num_steps: int,
    sigma_min: float = 0.01,
    sigma_max: float = 10.0,
    s: float = 0.008,
    pow: float = 2.0,
) -> jnp.ndarray:
    # Original t definition
    i = jnp.arange(num_steps + 1, dtype=jnp.float32)
    t = i / num_steps

    offset = 1 + s
    base = jnp.cos(0.5 * jnp.pi * (offset - t) / offset) ** pow
    sigmas = 0.5 * (sigma_max - sigma_min) * base + 0.5 * sigma_min

    # Flip to descending order
    sigmas = sigmas[::-1]

    return sigmas

def build_karras_sigmas(num_steps: int,
                        sigma_min: float,
                        sigma_max: float,
                        rho: float = 7.0) -> jnp.ndarray:
    """
    EDM/Karras sigma schedule (descending order):
    i = 0..N-1
    """
    i = jnp.arange(num_steps, dtype=jnp.float32)
    denom = jnp.maximum(num_steps - 1, 1)
    ramp = i / denom
    s_max = sigma_max ** (1.0 / rho)
    s_min = sigma_min ** (1.0 / rho)
    sigmas = (s_max + ramp * (s_min - s_max)) ** rho  # [sigma_max ... sigma_min]
    sigmas = jnp.concatenate([sigmas, jnp.array([0.0], dtype=jnp.float32)], axis=0)  # Length = N + 1 
    return sigmas

