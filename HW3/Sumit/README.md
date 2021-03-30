# Assignment 3
## Setup and contenets
* To perform the simulation, run include("startup.jl") and then include("panda.jl") which are located in the odes folder
* panda.jl includes the code for:
    * Joint angle trajectory generation using cubic interpolation
    * PD control without tracking 
    * PD control with tracking 
    * Computed Torque Control
    
## Cubic interpolation / cubic spline trajectory
* For this assignment I use cubic interpolation to generate the desired trajectory $q$, $\dot{q}$, and $\ddot{q}$.
* This ensures a smooth trajectory. 
* The time is scaled down to run from 0 to 1 (t_norm in the code).
* The joint angle, velocity, and acceleration are given by:
   * $q(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3$,
   * $\dot{q}(t) = a_1 t + a_2 t + a_3 t^2$
   * $\ddot{q}(t) = a_2 + a_3 t$
* Let the final velocities at t_norm = 0,1 be 0. Then by evaluating the above expressions of  joint angle, velocity, and acceleration at t=0 and t=1 we get
   * $a_0 = q_0$ (Initial position)
   * $a_1=0$
   * $a_2=1.5(q_d-q_0)$
   * $a_3=0.5(q_0-q_d)$

* These concepts were explained in EE 599 Introduction to Robotics. These concepts are discussed in "Intoduction to Robotics Mechanics and Control - Craig"

## Error norms and conclusion
The error norm in the case of :
* PD control without tracking is 0.00377.
* PD control with tracking is 0.003894
* Computed Torque Control is 0.00049

Generally, increasing $k_p$ tends to decrease the error, however care has to be taken to not keep $k_d$ too small as smaller k_d would lead to oscillations in the transitent response.

Acknowledgements:  I thank Landon and Huitao for assisting me with the general approach towards the assignment and debugging of the code
