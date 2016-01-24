module Cost

using .. Flimsy

export mse,
       categorical_cross_entropy,
       categorical_cross_entropy_with_scores

const CROSS_ENTROPY_EPS = 1e-20

include("cost/mse.jl")
include("cost/categorical_cross_entropy.jl")
include("cost/categorical_cross_entropy_with_scores.jl")

end