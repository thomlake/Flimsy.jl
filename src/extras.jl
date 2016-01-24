module Extras
    include("extras/zscore.jl")
    include("extras/logsumexp.jl")

    onehot(i::Int, d::Int) = (x = zeros(d); x[i] = 1; x)
end