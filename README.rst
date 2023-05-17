Multi-Agent Inverse Reinforcement Learning on a Particle Swarm
===============================================================

Note: In order to run CMA-ES and IMARL an installation of ``korali`` is required.

Hyperparameter optimization
---------------------------

To optimize the hyperparameter (radii zone of repulsion, zone of alignment, zone of atttraction) with CMA-ES run

.. code-block:: bash
    
    python run-cmaes.py


To optimize time-averaged rotation for a swarm of size 25 in 3 dimensions and trajectory length 1000 (default)

.. code-block:: bash
    
    python run-cmaes.py --N 25 --dim 3 --obj 0


Synthetic Data
---------------------------

To generate 50 trajectories of a swarm with 25 fish with 7 nearest neighbours and trajectory length 1000 in 3 dimensions run the command

.. code-block:: bash

    python generateTrajectories.py --N 25 --NT 1000 --NN 7 --D 3 --num 50

IMARL
---------------------------

To run IRL with synthetic data run

.. code-block:: bash

    python run-vracer-irl.py

Launch scripts for the hyperparameter search of IMARL can be found in ``_jobs``.

Use the following files for post-processing of the outputs:

.. code-block:: bash

    evaluateValue.py

    evaluateReward.py

    plotReward.py

    plotTrajectory.py

    makegif.py


