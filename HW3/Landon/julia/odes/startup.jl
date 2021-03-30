import Pkg;
Pkg.activate(@__DIR__);
using LinearAlgebra, StaticArrays
using RigidBodyDynamics, RigidBodySim
using MeshCat, MeshCatMechanisms
vis = Visualizer();open(vis)
using Gadfly, Cairo, Fontconfig
using PandaRobot
#import PandaRobot # for visualizing Panda

function display_urdf(urdfPath,vis)
    # Displays mechanism at config all zeros
    # urdfPath must be a string
    mechanism = parse_urdf(Float64,urdfPath)
    state = MechanismState(mechanism)
    zero_configuration!(state);
    mvis = MechanismVisualizer(mechanism, URDFVisuals(urdfPath),vis)
    manipulate!(state) do x
        set_configuration!(mvis, configuration(x))
    end
    for bd in bodies(mechanism)
        setelement!(mvis,default_frame(bd),0.5,"$bd")
    end
    return mvis, mechanism
end

# Example using Panda robot:
# urdfPath = PandaRobot.urdfpath()
# pandamech = display_urdf(urdfPath,vis)
# display_urdf("anymal.urdf",vis)
# display_urdf("panda.urdf",vis)
