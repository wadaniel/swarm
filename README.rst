Multi-Agent Inverse Reinforcement Learning on a Particle Swarm
===============================================================

Note: In order to run CMA-ES and IMARL an installation of ``korali`` is required.

Hyperparameter optimization
---------------------------
To optimize the hyperparameter (radii zone of repulsion, zone of alignment, zone of atttraction) with CMA-ES run

.. code-block:: bash
    
    python run-cmaes.py

Synthetic Data
---------------------------

To generate synthetic demonstration data run 

.. code-block:: bash

    python generateTrajectories.py

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


