
immutable FeedForward{F<:Activation,V<:Variable} <: Component{V}
    f::F
    w::Vector{V}
    b::Vector{V}
    function FeedForward(f::F, w::Vector{V}, b::Vector{V})
        length(w) == length(b) || error("w and b must have the same number of elements")
        m_prev = size(w[1], 1)
        size(b[1]) == (m_prev, 1) || throw(DimensionMismatch("Bad size(b[1]) != ($m_prev, 1)"))
        for i = 2:length(w)
            m, n = size(w[i])
            m_prev == n || throw(DimensionMismatch("Bad size(w[$i]) == ($m, $n) != ($m, $m_prev)"))
            size(b[i]) == (m, 1) || throw(DimensionMismatch("Bad size(b[$i]) != ($m, 1)"))
            m_prev = m
        end
        return new(f, w, b)
    end
end
FeedForward{F<:Activation,V<:Variable}(f::F, w::Vector{V}, b::Vector{V}) = FeedForward{F,V}(f, w, b)

function FeedForward{F<:Activation}(f::F, sz::Int...)
    dims = reverse(sz)
    depth = length(dims) - 1
    w = [rand(Normal(0, 0.01), dims[i+1], dims[i]) for i = 1:depth]
    b = [zeros(dims[i+1], 1) for i = 1:depth]
    return FeedForward(f=f, w=w, b=b)
end

FeedForward(sz::Int...) = FeedForward(Relu(), sz...)

@comp depth(params::FeedForward) = length(params.w)

@comp function feedforward(params::FeedForward, x::Variable)
    h = params.f(affine(params.w[1], x, params.b[1]))
    for i = 2:length(params.w)
        h = params.f(affine(params.w[i], h, params.b[i]))
    end
    return h
end
