# NIPS2015

Welcome to our anonymous NIPS submission. All code remains copyright of the anonymous authors, released under the GPL license (see LICENSE)

structure
--
There are two directories in this repo. `mcmcGP` contains a python module which implements the method. `experiments` contains subdirectories with code to replicate the examples in the paper. 

install
--
To work with our code, you'll need to instal our little module. We recommend running
    python setup.py build_ext --inplace
to build the cython modeules, and then add mcmcGP to your PYTHONPATH.

requisites
--
The code needs numpy, scipy, matplotlib, pandas, jug (https://github.com/luispedro/jug) and GPy (https://github.com/SheffieldML/GPy). The standard version of jug will suffice (`pip install jug`), but the bleeding edge of GPy is needed (see their install instructions). 

running
--
Running the code takes a long time. There are hundreds of models to fit! you'll need to use jug for image_demo, spatial_demo, etc. 

A good place to start is simple_classification, which should run in about a minute without jug. We recommend IPython for interacting with the code and figures.
