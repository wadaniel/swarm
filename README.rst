Multi-Agent Inverse Reinforcement Learning on a Particle Swarm
===============================================================

Hyperparameter optimization
---------------------------
To optimize the hyperparameter (radii zone of repulsion, zone of alignment, zone of atttraction) with CMA-ES run

.. code-block:: bash
    
    python run-cmaes.py

Synthetic Data
---------------------------

To generate synthetic demonstration data run 

.. code-block:: bash

    python generatrTrajectories.py

IMARL
---------------------------

To run IRL with synthetic data run

.. code-block:: bash

    python run-vracer-irl.py


Use the following files for post-processing of the results:

.. code-block:: bash

    evaluateValue.py

    evaluateReward.py

    plotReward.py

    plotTrajectory.py

    makegif.py


