
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
       SigmoidRegression


export SimpleRecurrent

abstract RecurrentComponent{T} <: Component{T}

include("components/value_component.jl")
include("components/linear_model.jl")
include("components/simple_recurrent.jl")

end