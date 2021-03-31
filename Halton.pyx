# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

# distutils: language = c++
import numbers

import numpy as np
cimport numpy as np

from numpy.random cimport BitGenerator, SeedSequence, PCG64


cpdef double [:] van_der_corput_cy(Py_ssize_t n, np.uint32_t base=2,
                                    Py_ssize_t start_index=0,
                                    scramble=False, rng=None):
    cdef:
        double [:] sequence = np.zeros(n, dtype=np.double)
        double [:] quotient = np.arange(start_index, start_index + n,
                                        dtype=np.double)
        double [:] remainder = np.zeros(n, dtype=np.double)
        double b2r = 1 / <double> base

    while (1 - b2r) < 1:
        for i in range(n):
            remainder[i] = quotient[i] % base
        if scramble:
            # permutation must be the same for all points of the sequence
            perm = rng.permutation(base)
            for i in range(n):
                remainder[i] = perm[int(remainder[i])]

        for i in range(n):
            sequence[i] += remainder[i] * b2r

        b2r /= base
        for i in range(n):
            quotient[i] = (quotient[i] - remainder[i]) / base

    return sequence