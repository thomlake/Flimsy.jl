
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

function logsumexp(x::AbstractFloat...)
    b = maximum(x)
    b == -Inf && return -Inf
    s = 0.0
    for xi in x
        s += exp(xi - b)
    end

    c = b + log(s)
    return c
end

