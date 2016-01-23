
function logsumexp{T<:AbstractFloat}(x::Array{T})
    b = maximum(x)
    b == -Inf && return -Inf
    s = 0.0
    for i in eachindex(x)
        @inbounds s += exp(x[i] - b)
    end

    c = b + log(s)
    return c
end

logsumexp{T<:AbstractFloat}(x::T...) = logsumexp(collect(x))
