
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
       CTCOutput,
       SimpleRecurrent,
       GatedRecurrent,
       LSTM

abstract RecurrentComponent{T} <: Component{T}

include("components/constructor.jl")
include("components/value_component.jl")
include("components/linear_model.jl")
include("components/ctc_output.jl")
include("components/simple_recurrent.jl")
include("components/gated_recurrent.jl")
include("components/lstm.jl")

end