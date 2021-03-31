"""Quasi-Monte Carlo engines and helpers."""
import copy
import numbers
from abc import ABC, abstractmethod
import math
import warnings

import numpy as np

import scipy.stats as stats
from Halton import van_der_corput_cy

__all__ = ['QMCEngine', 'Halton',]


# Based on scipy._lib._util.check_random_state
def check_random_state(seed=None):
    if seed is None or isinstance(seed, (numbers.Integral, np.integer)):
        if not hasattr(np.random, 'Generator'):
            # This can be removed once numpy 1.16 is dropped
            msg = ("NumPy 1.16 doesn't have Generator, use either "
                   "NumPy >= 1.17 or `seed=np.random.RandomState(seed)`")
            raise ValueError(msg)
        return np.random.default_rng(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    elif isinstance(seed, np.random.Generator):
        # The two checks can be merged once numpy 1.16 is dropped
        return seed
    else:
        raise ValueError('%r cannot be used to seed a numpy.random.Generator'
                         ' instance' % seed)


def van_der_corput(n, base=2, start_index=0, scramble=False, seed=None):
    rng = check_random_state(seed)

    return van_der_corput_cy(n, base, start_index, scramble, rng)


def primes_from_2_to(n):
    """Prime numbers from 2 to *n*.

    Taken from [1]_ by P.T. Roy, licensed under
    `CC-BY-SA 4.0 <https://creativecommons.org/licenses/by-sa/4.0/>`_.

    Parameters
    ----------
    n : int
        Sup bound with ``n >= 6``.

    Returns
    -------
    primes : list(int)
        Primes in ``2 <= p < n``.

    References
    ----------
    .. [1] `StackOverflow <https://stackoverflow.com/questions/2068372>`_.

    """
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=bool)
    for i in range(1, int(n ** 0.5) // 3 + 1):
        k = 3 * i + 1 | 1
        sieve[k * k // 3::2 * k] = False
        sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


def n_primes(n):
    """List of the n-first prime numbers.

    Parameters
    ----------
    n : int
        Number of prime numbers wanted.

    Returns
    -------
    primes : list(int)
        List of primes.

    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
              61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
              131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
              197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269,
              271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
              353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431,
              433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
              509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599,
              601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673,
              677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761,
              769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857,
              859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947,
              953, 967, 971, 977, 983, 991, 997][:n]

    if len(primes) < n:
        big_number = 2000
        while 'Not enough primes':
            primes = primes_from_2_to(big_number)[:n]
            if len(primes) == n:
                break
            big_number += 1000

    return primes


class QMCEngine(ABC):
    @abstractmethod
    def __init__(self, d, seed=None):
        self.d = d
        self.rng = check_random_state(seed)
        self.rng_seed = copy.deepcopy(seed)
        self.num_generated = 0

    @abstractmethod
    def random(self, n=1):
        """Draw `n` in the half-open interval ``[0, 1)``.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space.
            Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            QMC sample.

        """
        # self.num_generated += n

    def reset(self):
        """Reset the engine to base state.

        Returns
        -------
        engine : QMCEngine
            Engine reset to its base state.

        """
        seed = copy.deepcopy(self.rng_seed)
        self.rng = check_random_state(seed)
        self.num_generated = 0
        return self

    def fast_forward(self, n):
        """Fast-forward the sequence by `n` positions.

        Parameters
        ----------
        n : int
            Number of points to skip in the sequence.

        Returns
        -------
        engine : QMCEngine
            Engine reset to its base state.

        """
        self.random(n=n)
        return self


class Halton(QMCEngine):
    def __init__(self, d, scramble=True, seed=None):
        super().__init__(d=d, seed=seed)
        self.seed = seed
        self.base = n_primes(d)
        self.scramble = scramble

    def random(self, n=1):
        """Draw `n` in the half-open interval ``[0, 1)``.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            QMC sample.

        """
        # Generate a sample using a Van der Corput sequence per dimension.
        # important to have ``type(bdim) == int`` for performance reason
        sample = [van_der_corput(n, int(bdim), self.num_generated,
                                 scramble=self.scramble,
                                 seed=copy.deepcopy(self.seed))
                  for bdim in self.base]

        self.num_generated += n
        return np.array(sample).T.reshape(n, self.d)
