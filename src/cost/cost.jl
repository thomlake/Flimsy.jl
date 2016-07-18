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

include("mse.jl")
# include("categorical_cross_entropy.jl")
# include("categorical_cross_entropy_with_scores.jl")
# include("bernoulli_cross_entropy.jl")
# include("bernoulli_cross_entropy_with_scores.jl")
# include("ctc_with_scores.jl")
# # include("cost/reinforce.jl")

end