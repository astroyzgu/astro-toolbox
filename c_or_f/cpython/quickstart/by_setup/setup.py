# from distutils.core import setup
from setuptools import setup, find_packages, find_namespace_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

#setup(
#      ext_modules = cythonize(["helloworld.pyx", "mpi4pytest.pyx", "cmpi4pytest.pyx", "hello.cpp"])
#)

ext_cpp = Extension(name='hello',
                  sources=['hello.cpp'],
                  language='c++', 
                  extra_compile_args=["-std=c++11"],
                  cython_directives=dict(embedsignature=True),
)

setup( ext_modules = [ext_cpp,], )  
