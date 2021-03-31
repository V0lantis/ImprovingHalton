import numpy as np


#pythran export dprod(int list, int list)
def dprod(arr0, arr1):
    return sum([x*y for x,y in zip(arr0, arr1)])


#pythran export van_der_corput_internal(int, int, int, bool, int -> int)
def van_der_corput_internal(n, base, start_index, scramble, seed):
    print("hello world")
    # rng = check_random_state(seed)
    # sequence = np.zeros(n)
    #
    # quotient = np.arange(start_index, start_index + n)
    # b2r = 1 / base
    #
    # while (1 - b2r) < 1:
    #     remainder = quotient % base
    #
    #     if scramble:
    #         # permutation must be the same for all points of the sequence
    #         perm = rng.permutation(base)
    #         remainder = perm[np.array(remainder).astype(int)]
    #
    #     sequence += remainder * b2r
    #     b2r /= base
    #     quotient = (quotient - remainder) / base
    #
    # return sequence
