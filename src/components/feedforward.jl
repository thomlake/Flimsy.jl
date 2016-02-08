
immutable FeedForward{V<:Variable} <: Component{V}
    f::Function
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

function FeedForward(f::Function, sz::Int...)
    dims = reverse(sz)
    depth = length(dims) - 1
    w = [rand(Normal(0, 0.01), dims[i+1], dims[i]) for i = 1:depth]
    b = [zeros(dims[i+1], 1) for i = 1:depth]
    return FeedForward(f=f, w=w, b=b)
end

FeedForward(sz::Int...) = FeedForward(relu, sz...)

@component function feedforward{C<:FeedForward}(params::C, h::Variable)
    for i = 1:length(params.w)
        h = params.f(affine(params.w[i], h, params.b[i]))
    end
    return h
end

# SimpleRecurrent(m::Int, n::Int) = SimpleRecurrent(tanh, m, n)

# @component Base.step(p::SimpleRecurrent, x::Variable, htm1::Variable) = p.f(plus(linear(p.w, x), linear(p.u, htm1), p.b))

# @component Base.step(p::SimpleRecurrent, x::Variable) = step(p, x, p.h0)

# @component function unfold{T<:Variable}(p::SimpleRecurrent, x::Vector{T})
#     h = Array(vartype(T), length(x))
#     h[1] = step(p, x[1])
#     for t = 2:length(x)
#         h[t] = step(p, x[t], h[t-1])
#     end
#     return h
# end
