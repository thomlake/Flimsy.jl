
immutable SimpleRecurrent{V<:Variable} <: RecurrentComponent1{V}
    f::Function
    w::V
    u::V
    b::V
    h0::V
    function SimpleRecurrent(f::Function, w::V, u::V, b::V, h0::V)
        m, n = size(w)
        size(u) == (m, m) || throw(DimensionMismatch("Bad size(u) == $(size(u)) != ($m, $m)"))
        size(b) == (m, 1) || throw(DimensionMismatch("Bad size(b) == $(size(b)) != ($m, 1)"))
        size(h0) == (m, 1) || throw(DimensionMismatch("Bad size(h0) == $(size(h0)) != ($m, 1)"))
        return new(f, w, u, b, h0)
    end
end
SimpleRecurrent{V<:Variable}(f::Function, w::V, u::V, b::V, h0::V) = SimpleRecurrent{V}(f, w, u, b, h0)

SimpleRecurrent(f::Function, m::Int, n::Int) = SimpleRecurrent(
    f=f,
    w=orthonormal(f, m, n),
    u=orthonormal(f, m, m),
    b=zeros(m, 1),
    h0=zeros(m, 1),
)

SimpleRecurrent(m::Int, n::Int) = SimpleRecurrent(tanh, m, n)

@component Base.step(p::SimpleRecurrent, x::Variable, htm1::Variable) = p.f(plus(linear(p.w, x), linear(p.u, htm1), p.b))

@component Base.step(p::SimpleRecurrent, x::Variable) = step(p, x, p.h0)

@component function unfold{T<:Variable}(p::SimpleRecurrent, x::Vector{T})
    h = Array(vartype(T), length(x))
    h[1] = step(p, x[1])
    for t = 2:length(x)
        h[t] = step(p, x[t], h[t-1])
    end
    return h
end
