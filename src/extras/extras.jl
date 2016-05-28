module Extras
    include("zscore.jl")
    include("logsumexp.jl")

    function onehot(i::Int, d::Int)
        x = zeros(d) 
        x[i] = 1
        return x
    end

    function bagofwords{T}(::Type{T}, I::Vector{Int}, d::Int)
        x = zeros(T, d)
        for i in I
            x[i] = 1
        end
        return x
    end

    bagofwords(I::Vector{Int}, d::Int) = bagofwords(Float64, I, d)
end