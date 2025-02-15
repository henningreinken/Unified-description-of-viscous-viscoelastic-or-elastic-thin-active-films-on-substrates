# Unified-description-of-viscous-viscoelastic-or-elastic-thin-active-films-on-substrates
This repository contains Python scripts used to investigate the dynamics in viscous, viscoelastic, or elastic thin active films.

<h2>REQUIREMENTS</h2>

Python 3

tested with Python 3.10.12


<h2>USAGE INSTRUCTIIONS</h2>

All scripts are executed via the command line from within the main folder. For simplicity, parameter values are set directly within the scripts.

The scripts starting "*simulation...*" are used for the main calculations. They solve the evolution equations for the polar order parameter field, the velocity field and the displacement field via a pseudo spectral method and perform additional analysis of the dynamics.

*simulation_hysteresis_generateInitialValues.py* <br>
This script always has to be executed first to generate the initial values for subsequent calculations. It starts with a given value of the active force strength, determines the spatiotemporal dynamics and saves the resulting fields. Then, the active force strength is decreased by a small amount and the calculation is performed again. These steps are repeated until a given value of active force strength is reached.

*simulation_continue.py* <br>
This script continues the calculations for a particular active force strength starting from the initial values determined before via *simulation_hysteresis_generateInitialValues.py*. It also saves system-averaged quantities.

*FourierAnalysis.py* <br>
This script performs a Fourier analysis of the system-averaged quantities determiend via *simulation_continue.py* in order to find the dominant frequency of global rotations of the fields, their magnitudes, as well as their phase shifts.
