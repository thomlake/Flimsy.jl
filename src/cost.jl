module Cost

using .. Flimsy

export mse,
       categorical_cross_entropy,
       categorical_cross_entropy_with_scores,
       bernoulli_cross_entropy,
       bernoulli_cross_entropy_with_scores,
       ctc_with_scores,
       reinforce

const CROSS_ENTROPY_EPS = 1e-20

include("cost/mse.jl")
include("cost/categorical_cross_entropy.jl")
include("cost/categorical_cross_entropy_with_scores.jl")
include("cost/bernoulli_cross_entropy.jl")
include("cost/bernoulli_cross_entropy_with_scores.jl")
include("cost/ctc_with_scores.jl")
# include("cost/reinforce.jl")

end