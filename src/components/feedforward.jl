
immutable FeedForward{F<:Activation} <: Component
    f::F
    w::Vector{GradVariable}
    b::Vector{GradVariable}
    function FeedForward(f::F, w::Vector{GradVariable}, b::Vector{GradVariable})
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

FeedForward{F<:Activation}(f::F, w::Vector{GradVariable}, b::Vector{GradVariable}) = FeedForward{F}(f, w, b)

function FeedForward{F<:Activation}(f::F, sz::Int...)
    dims = reverse(sz)
    depth = length(dims) - 1
    w = [rand(Normal(0, 0.01), dims[i+1], dims[i]) for i = 1:depth]
    b = [zeros(dims[i+1], 1) for i = 1:depth]
    return FeedForward(f=f, w=w, b=b)
end

FeedForward(sz::Int...) = FeedForward(Relu(), sz...)

depth(params::FeedForward) = length(params.w)

depth(scope::Scope, params::FeedForward) = depth(params.w)

function feedforward(scope::Scope, params::FeedForward, h::Variable)
    @with scope begin
        for i = 1:length(params.w)
            h = activate(params.f, affine(params.w[1], h, params.b[1]))
        end
        return h
    end
end
