module Cost

using .. Flimsy

export mse,
       categorical_cross_entropy,
       categorical_cross_entropy_with_scores

include("cost/mse.jl")
include("cost/categorical_cross_entropy.jl")
include("cost/categorical_cross_entropy_with_scores.jl")

end