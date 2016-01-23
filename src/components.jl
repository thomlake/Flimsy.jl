
module Components

using .. Flimsy

export score,
       predict,
       probs,
       cost

export ValueComponent,
       LinearModel,
       LinearRegression,
       LogisticRegression

include("components/value_component.jl")
include("components/linear_model.jl")

end