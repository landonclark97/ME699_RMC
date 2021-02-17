# Robot Modeling & Control

HW1:
* located in the julia/odes directory. all code is in startup.jl.
* the robot is a 10 dof, 2 end-effector robot modeled after the human upper body.
* there are no joint constraints implemented in this project, so collisions are possible and common.
* to use the program, open Julia and include startup.jl. the program will prompt the user to input x, y, and z coordinates for a desired point. once selected, the closest end-effector will be used to move towards this point. once a point has been reached, a constraint will be placed on the joints shared by both end-effectors. The user can then decide whether or not to continue after moving to each point. 
