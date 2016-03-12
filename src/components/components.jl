
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
       VectorComponent,
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

include("constructor.jl")
include("common.jl")
include("value_component.jl")
include("linear_model.jl")
include("ctc_output.jl")
include("feedforward.jl")
include("simple_recurrent.jl")
include("gated_recurrent.jl")
include("lstm.jl")

end