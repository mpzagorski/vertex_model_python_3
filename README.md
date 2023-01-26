# Vertex model of growing neural tube

The following python 3 implementation was used in 

> Bocanegra L, Singh A, Hannezo E, Zagorski M, Kicheva A. Cell cycle dynamic control fluidity of the developing mouse neuroepithelium (2023). (accepted for publication in *Nature Physics*)

This implementation was based on the python 2 implementation published along

> Guerrero P, Perez-Carrasco R, Zagorski M, Page D, Kicheva A, Briscoe J, Page KM. Neuronal differentiation influences progenitor arrangement in the vertebrate neuroepithelium (2019). *Development*, **146**(23) [https://doi.org/10.1242/dev.176297](https://doi.org/10.1242/dev.176297)

This python 2 implementation is available at [original code](https://bitbucket.org/Pigueco/vertex_model_python_2.7/src/master/).


### Functionality

The code is used to simulate 2D layer of cells represented as polygons. The layer of cells has periodic boundary conditions corresponding to toroidal geometry. By including external drag coefficients the anisotropic growth that takes place in the developing spinal cord can be simulated. In comparison with the original implementation (Guerrero et al., 2019), the current code includes noise in the internal junctional line tension. 


### Requirements

* Python 3.7 (was tested with Python 3.8.16)
* numpy (was tested with numpy 1.23.5)
* scipy (was tested with scipy 1.9.3)


### How to run it?

This code was restructured to be suitable for running from the command line, but it is also possible to explore its functionality by modifying parameters in ‘run_main_simulation.py’. The file ‘Global_Constant.py’ can be also used to modify some of the parameters, yet subset of these parameters can be overridden by specification through command line or ‘run_main_simulation.py’. The parameters used in the actual simulation are exported to log files.

Example of a command line use that produces output files present in the OUT folder:

```bash
python main_run_simulation.py 1.0 0.12 -0.074 0 0.02 0.01 50 0 100 11 21 10 0.1282 0.02 0.45 12345 ./OUT
```

The description of the parameters used can be found in ‘run_main_simulation.py’.

### Possible issues

If after installing all relevant packages, the execution of the code raises the following type of error

```python
ValueError: When changing to a larger dtype, its size must be a divisor of the total size in bytes of the last axis of the array.
```

It is very likely that uncommenting the line 50 and commenting out the line 49 in ‘initialisation.py’ will solve this issue. In this case the representation of integer types in the code needs to match the system representation, in this case ‘'formats': ['i8']*6’ goes to ‘'formats': ['i']*6’.

### Updates and code development

The code is developed and maintained by Marcin Zagórski. For updates please contact me directly at marcin.zagorski@uj.edu.pl or visit my group webpage [https://zagorskigroup.com/](https://zagorskigroup.com/).
