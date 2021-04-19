q_0 = [0.01;-0.5;-0.0;-2.0;-0.3;1.5;-0.7;0.1;0.1]
q_d = [0.0;0.0;0.0;0.0;0.0;pi;0.01;0.01;0.01]

# Question 1, get the traj of q_t,q_dt,q_ddt. With cubic interpolation
function traj(time)
    # compute the desired joint angle at time t with cubic interpolation
    # a0 = q_0, a1=0,a2=3/2*(q_d-q_0),a3=1/2(q_0-q_d)
    t = time/10
    q_t = q_0+3/2*(q_d-q_0)*t^2+1/2*(q_0-q_d)*t^3
    q_dt = 3*(q_d-q_0)*t+3/2*(q_0-q_d)*t^2
    q_ddt = 3*(q_d-q_0)+3*(q_0-q_d)*t
    return q_t,q_dt,q_ddt
end

# Question2, control_PD!
function control_PD!(τ, t, state)
    kp = diagm([10000,10000, 10000, 10000, 10000, 10000, 10000,10000,10000])
    kd = diagm([1,1, 1, 1, 1, 1, 1,1,1])
    q_t,q_dt,q_ddt = traj(t)
    # Compute a value for τ
    τ .= -kd * (velocity(state) - q_dt) - kp*(configuration(state) - q_t) #why negative???
    # Saturate
    act_sat = 50; # Actuator limits
    τ .= map( x -> x > act_sat ? act_sat : x,τ)
    τ .= map( x -> x < -act_sat ? -act_sat : x,τ)
end

# Question3, control_CTC!
function control_CTC!(τ, t, state)
    # proportion to the PD control
    P = 0.4
    kp = P*diagm([10000,10000, 10000, 10000, 10000, 10000, 10000,10000,10000])
    kd = 10*diagm([1,1,1,1,1,1,1,1,1])
    q_t,q_dt,q_ddt = traj(t)
    # Compute a value for τ
    τ .= RigidBodyDynamics.mass_matrix(state)*(kp * (q_t - configuration(state)) + kd * (q_dt-velocity(state)) + q_ddt)+RigidBodyDynamics.dynamics_bias(state)
    # Saturate
    act_sat = 50; # Actuator limits
    τ .= map( x -> x > act_sat ? act_sat : x,τ)
    τ .= map( x -> x < -act_sat ? -act_sat : x,τ)
end

p = plot()
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


# Refresh visualization
delete!(vis)
# Load mechanism info
urdfPath = "panda.urdf"
mvis, mechanism = display_urdf(urdfPath,vis)
# Create state and set initial config and velocity
state = MechanismState(mechanism)
set_configuration!(state,q_0)
zero_velocity!(state)
# Update mechanism visual
set_configuration!(mvis, configuration(state))



# Define ODE Problem, which defines closed loop using  control_PD!
problem = ODEProblem(Dynamics(mechanism,control_PD!), state, (0., 10.));
# Solve ODE problem using Tsit5 scheme, and given numerical tolerances
sol = solve(problem, Tsit5(),reltol=1e-8,abstol=1e-8);
# Animate solution
setanimation!(mvis, sol; realtime_rate = 1.0);
# Plot joint angles vs time using Gadfly
plot_sol(p,sol,[colorant"red"],false,"foo.pdf")
for i in 1:7
    println("final joint angle $i : $(sol[end][i])")
end

# get the configuration error norm
print("Final configuration norm : $(norm(sol[end][1:7]-q_d[1:7]))")

# Show plot in browser
p
