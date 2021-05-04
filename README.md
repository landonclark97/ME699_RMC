# Robot Modeling & Control

## HW1:
* located in the HW1/julia/odes directory. all code is in startup.jl.
* the robot is a 10 dof, 2 end-effector robot modeled after the human upper body.
* there are no joint constraints implemented in this project, so collisions are possible and common.
* to use the program, open Julia and include startup.jl. the program will prompt the user to input x, y, and z coordinates for a desired point. once selected, the closest end-effector will be used to move towards this point. once a point has been reached, a constraint will be placed on the joints shared by both end-effectors. The user can then decide whether or not to continue after moving to each point. 

## HW2:
* located in HW2 directory.
* contains solution for mass matrix and conservation force vector, still working on deriving the Coriolis matrix.
* provides function custom_inversedynamics, inputs need to be column vectors.

## HW3:
* located in the HW3 directory.
* As this homework was to be done individually, each of our members has a folder for himself.

## Paper Reviews:
* round one paper reviews can be found in the corresponding folder, with a subfolder for each member of our group.

## Final Project, SEA Controller:
* project aimed at controlling a SEA robot with reinforcement learning algorithms.
* final project code can be found in the sea_controller directory.
