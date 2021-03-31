from distutils.core import setup
from distutils.extension import Extension

import numpy as np
from Cython.Build import cythonize
from Cython.Distutils import build_ext


ext_modules = [
    Extension(
        name="Halton",
        sources=["Halton.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="Halton",
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(ext_modules),
)
