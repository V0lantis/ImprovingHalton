import timeit
import sys

import numpy as np

from _qmc import Halton
from _qmc_cy import Halton as Halton_cy


def cy_hatlon_wrapper(dim, n):
    sampler = Halton_cy(d=dim, scramble=False)
    sample = sampler.random(n=n)

def scipy_halton_wrapper(dim, n):
    sampler = Halton(d=dim, scramble=False)
    sample = sampler.random(n=n)


FUNS = [
    ("scipy", "scipy_halton_wrapper"),
    ("cython", "cy_hatlon_wrapper"),
    # ("cython", "discrepancy_cython"),
    # ("cython threaded (workers=0)", "discrepancy_cython_threaded0"),
    # ("cython threaded (workers=8)", "discrepancy_cython_threaded8"),
]

np.random.seed(0)

TEST_VALUES = [
    (100, 2),
    (100, 10),
    # (1000, 10),
    # (1000, 20),
    # (10000, 100),
]

NREPEAT = 7
EFFORT = 1


def ct(s):
    if s > 1:
        return f"{s:4.1f}s"
    for p, u in ((3, "ms"), (6, "µs")):
        if s * 10 ** p > 1:
            return f"{s*10**p:4.1f}{u}"
    return f"{s*10**9:4.1f}ns"


method_funs_times = {}
method_funs_values = {}

funs_times = []
funs_values = []

for funname, fun in FUNS:
    times = []
    values = []
    for i, sample in enumerate(TEST_VALUES):
        d, n = sample
        nnumber = int(EFFORT/(1e-4*(n**2*d)))+1
        times.append(
            np.array(
                timeit.repeat(
                    f"{fun}({d}, {n})",
                    number=nnumber,
                    repeat=NREPEAT,
                    setup=f"from __main__ import {fun}",
                )
            ).mean()
            / nnumber
        )
        value = 0.
        # enable to check value
        # exec(f"value={fun}(SAMPLES[{i}], method={method!r})")
        values.append(value)

    print(f"| {funname} | " + " | ".join(f"{ct(t)}" for t in times) + " |")
    funs_times.append(times)
    funs_values.append(values)

for (times, (funname, _)) in zip(funs_times, FUNS):
    print(
        f"| [scipy] / [{funname}] | "
        + " | ".join(f"{b/a:.1f}" for a, b in zip(times, funs_times[0]))
        + " |"
    )
print()