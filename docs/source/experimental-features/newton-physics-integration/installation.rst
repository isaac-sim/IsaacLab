Installation
============= 

Installing the Newton Physics Integration Branch requires three things:

1) Isaac sim 5.0 or greater
2) The experimental branch
3) Rebuilding Isaac Lab

To begin, verify the version of Isaac Sim by checking the title of the window created when launching the simulation app.  Alternatively, you can 
find more explicit version information under the ``Help -> About`` menu within the app. So long as that version is 5.0 or greater you will be able to  
use the experimental feature branch for Newton.  If your version is less than 5.0, you must first `update or reinstall Isaac Sim <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/quick-install.html>`_ before 
you can proceed further.

Next, navigate to the root directory of your local copy of the Isaac Lab repository and open a terminal.  Before we checkout the branch, we want to make sure we save all of our work (if we choose) and checkout the main branch first.
Begin by checking the status of the current branch 

.. code-block:: shell

    $> git status

If you are ready to checkout the experimental branches, this is what you should see

.. code-block:: shell

    On branch main
    Your branch is up to date with 'origin/main'.

    nothing to commit, working tree clean

If you see anything else, we strongly recommend you stash or commit your changes in order to avoid merge conflicts with the new branch
   

