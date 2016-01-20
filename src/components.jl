
module Components

using .. Flimsy

export score,
       probs,
       cost

export ValueComponent,
       SoftmaxRegression

include("components/value_component.jl")
include("components/softmax_regression.jl")

end