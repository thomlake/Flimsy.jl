module Extras
    include("zscore.jl")
    include("logsumexp.jl")

    onehot(i::Int, d::Int) = (x = zeros(d); x[i] = 1; x)
end