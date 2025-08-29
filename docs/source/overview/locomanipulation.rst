.. _locomanipulation:

Locomanipulation Data Generation
================================

This tutorial demonstrates how you can generate diverse locomanipulation
trajectories from static manipulation recordings.  

The overall workflow is as follows

1. Record a static manipulation trajectory of "picking up" and "dropping off" an object.
   In this phase, the robot base is stationary.  This is done by human teleoperation.

2. Augment the static manipulation trajectory using mimic data generation pipeline.  This will
   Create a diverse set of augmented static manipulation trajectories.

3. Run the "disjoint navigation" locomanipulation data generation pipeline to create
   end-to-end locomanipulation trajectories by combining the static manipulation sequences with 
   path planning.

Step 1 - Static manipulation teleoperation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, record a static manipulation trajectory with the G1 via teleoperation.

You may skip to step (2) using the following recording: ...

TODO:  reference manipulation teleoperation and provide pre-recorded teleop dataset

Step 2 - Mimic data augmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Second, augment the static manipulation trajectory by running the mimic pipeline.

TODO: reference mimic data generation and provide pre-recorded mimic dataset


Step 3 - Locomanipulation data generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Third, run the disjoint navigation replay data generation script to generate end-to-end
locomanipulation data from the static recordings.

If you are using the dataset generated above, you can use the following command.

.. code:: bash

    ./isaaclab.sh -p \
        scripts/imitation_learning/disjoint_navigation/replay.py \
        --device cpu \
        --kit_args="--enable isaacsim.replicator.mobility_gen" \
        --dataset="datasets/dataset_generated_g1_locomotion_teacher.hdf5" \
        --num_runs=1 \
        --lift_step=50 \
        --navigate_step=100 \
        --output_dir=datasets \
        --output_file_name=dataset_generated_disjoint_nav.hdf5


If you are using a different trajectory, you will need to change some parameters.  Notably, you will need to set

- --lift_step - The step index where the robot has finished grasping the object and is ready to lift it
- --navigate_step - The step index where the robot has finished lifting the object and is ready to navigate

These values can be determined empirically, by running the script above and observing the behavior.

Step 4 - Visualize the data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, to verify the data has been generated correctly, we can run a script to visualize the trajectory.

To do this you can run

.. code:: bash

    ./isaaclab.sh -p \
        scripts/imitation_learning/disjoint_navigation/display_dataset.py \
        --dataset="datasets/dataset_generated_g1_disjoint_navigation.hdf5" \
        --output_dir="datasets/dataset_generated_g1_disjoint_navigation_plots"

After this step, the folder datasets/dataset_generated_g1_disjoint_navigation_plots will show the output visualizations.
You can open this folder to browse the resulting images.
