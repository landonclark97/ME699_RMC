# Final time of simulation. Assume initial time is 0.
final_time = 10;


function traj(t::Float64)
    
    # Here we generate a cubic spline trajectory which is normalized for time t_norm=0 to t_norm=1
    # This is done to obtain a smooth trajectory

    t_norm =t/final_time;

    # compute the desired joint angle at time t using cubic splines / cubic interpolation 
    
    a_0 = q0
    a_1 = 0
    a_2 = 1.5*(qd-q0)
    a_3 = 0.5*(q0-qd)
    
    # Desired angle, velocity, and acceleration at time t 
    q_d = a_0 + a_2 * t_norm^2 + a_3 * t_norm^3
    qdot_d= 2 * a_2 * t_norm + 3* a_3 * t_norm^2
    qddot_d= 2*a_2 + 6*a_3*t_norm
    return q_d, qdot_d, qddot_d
end


    
function control_PD!(τ, t, state)
    q_d, qdot_d, qddot_d = traj(t)

      # Proportional and derivative control 

    k_p = 100*diagm([100,100, 100, 100, 100, 100, 100,100,100])
    
    k_d = 5*diagm([20,20, 20, 20, 20, 20, 20,20,20]) 
    # Compute a value for τ

    τ .= -k_d* velocity(state) - k_p * (configuration(state)-q_d)

    # Saturate
    act_sat = 50 # Actuator limits
    τ .= map( x -> x > act_sat ? act_sat : x,τ)
    τ .= map( x -> x < -act_sat ? -act_sat : x,τ)
end

function control_PD_TT!(τ, t, state)

    q_d, qdot_d, qddot_d = traj(t)
    # Proportional and derivative control 

    k_p = 200*diagm([100,100, 100, 100, 100, 100, 100,100,100])
    
    k_d = 1*diagm([20,20, 20, 20, 20, 20, 20,20,20]) 
    # Compute a value for τ

    τ .= -k_d * (velocity(state)-qdot_d) - k_p * (configuration(state)-q_d)

    # Saturate
    act_sat = 50 # Actuator limits
    τ .= map( x -> x > act_sat ? act_sat : x,τ)
    τ .= map( x -> x < -act_sat ? -act_sat : x,τ)
end

function control_CTC!(τ, t, state)

    # Proportional and derivative control 
    q_d, qdot_d, qddot_d = traj(t)

    # Proportional and derivative control 

    k_p = 150*diagm([100,100, 100, 100, 100, 100, 100,100,100])
    
    k_d = 0.1*diagm([20,20, 20, 20, 20, 20, 20,20,20]) 

    # Compute a value for τ
    aq= qddot_d + k_p * (q_d-configuration(state)) + k_d * (qdot_d-velocity(state));
    
    #Torque
    τ .= mass_matrix(state)*aq + dynamics_bias(state)
    # Saturate
    act_sat = 50; # Actuator limits
    τ .= map( x -> x > act_sat ? act_sat : x,τ)
    τ .= map( x -> x < -act_sat ? -act_sat : x,τ)
end

function plot_sol(p,sol,colorarg,saveflag,savename)
    qsol = vcat(sol[:]'...)
    for i=1:7
        push!(p,layer(x=sol.t,y=qsol[:,i],Geom.line,color=colorarg))
    end
    p
    if saveflag
        p |> PDF(savename)
    end
end

q0 = [0.01;-0.5;-0.0;-2.0;-0.3;1.5;-0.7;0.1;0.1] # initial joint angles
qd = [0.0;0.0;0.0;0.0;0.0;pi;0.01;0.01;0.01]     # final desired joint angles

function sim_control(control)
    
    # Refresh visualization
    delete!(vis)
    
    # Load mechanism info
    urdfPath = "panda.urdf" 
    mvis, mechanism = display_urdf(urdfPath,vis)
    
    # Create state and set initial config and velocity
    state = MechanismState(mechanism)
    set_configuration!(state,q0)
    zero_velocity!(state)
    
    # Update mechanism visual
    set_configuration!(mvis, configuration(state))

    p=plot();
    
    # Define ODE Problem, which defines closed loop using  control!
    problem = ODEProblem(Dynamics(mechanism,control), state, (0., final_time));
    
    # Solve ODE problem using Tsit5 scheme, and given numerical tolerances
    sol = solve(problem, Tsit5(),reltol=1e-8,abstol=1e-8);

    # Animate solution
    setanimation!(mvis, sol; realtime_rate = 1.0);
    
    # Plot joint angles vs time using Gadfly
    plot_sol(p,sol,[colorant"red"],false,nothing)

    p

    qs =zeros(9); #values of the joint angles from simulation

    for i in 1:9
        qs[i] = sol[end][i];
        println("final joint angle $i : $(sol[end][i])")
    end
    println("error norm : ",norm(qs-qd))
    println("\n")

end

# Simulate different control strategies

println("PD control without trajectory tracking \n")
sim_control(control_PD!)
sleep(2*final_time)


println("PD control with trajectory tracking \n")
sim_control(control_PD_TT!)
sleep(2*final_time)

println("Computed track Control \n")
sim_control(control_CTC!)


