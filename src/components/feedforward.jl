
immutable FeedForward{V<:Variable} <: Component{V}
    w::Vector{V}
    b::Vector{V}
    function FeedForward(f::Function, w::Vector{V}, b::Vector{V})
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
FeedForward{V<:Variable}(w::Vector{V}, b::Vector{V}) = 
    FeedForward{V}(f, w, b)

function FeedForward(sz::Int...)
    dims = reverse(sz)
    depth = length(dims) - 1
    w = [rand(Normal(0, 0.01), dims[i+1], dims[i]) for i = 1:depth]
    b = [zeros(dims[i+1], 1) for i = 1:depth]
    return FeedForward(w=w, b=b)
end

FeedForward(sz::Int...) = FeedForward(sz...)

@comp depth(params::FeedForward) = length(params.w)

@comp function feedforward(params::FeedForward, h::Variable)
    for i = 1:length(params.w)
        h = relu(affine(params.w[i], h, params.b[i]))
    end
    return h
end
