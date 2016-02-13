
module Components

using .. Flimsy
using HDF5: HDF5Group, attrs

export score,
       predict,
       probs,
       unfold,
       cost,
       feedforward

export RecurrentComponent,
       RecurrentComponent1,
       RecurrentComponent2

export ValueComponent,
       EmptyComponent

export LinearModel,
       LinearRegression,
       SoftmaxRegression,
       SigmoidRegression,
       CtcOutput,
       FeedForward,
       SimpleRecurrent,
       GatedRecurrent,
       Lstm

abstract RecurrentComponent{T} <: Component{T}
abstract RecurrentComponent1{T} <: RecurrentComponent{T}
abstract RecurrentComponent2{T} <: RecurrentComponent{T}

include("components/constructor.jl")
include("components/value_component.jl")
include("components/linear_model.jl")
include("components/ctc_output.jl")
include("components/feedforward.jl")
include("components/simple_recurrent.jl")
include("components/gated_recurrent.jl")
include("components/lstm.jl")

end