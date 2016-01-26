
module Components

using .. Flimsy

export score,
       predict,
       probs,
       unfold,
       cost

export RecurrentComponent

export ValueComponent

export LinearModel,
       LinearRegression,
       SoftmaxRegression,
       SigmoidRegression,
       SimpleRecurrent,
       GatedRecurrent

abstract RecurrentComponent{T} <: Component{T}

include("components/constructor.jl")
include("components/value_component.jl")
include("components/linear_model.jl")
include("components/simple_recurrent.jl")
include("components/gated_recurrent.jl")

end