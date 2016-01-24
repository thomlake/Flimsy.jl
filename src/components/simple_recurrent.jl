
immutable SimpleRecurrent{V<:Variable} <: RecurrentComponent{V}
    f::Function
    w::V
    u::V
    b::V
    h0::V
end

SimpleRecurrent(f::Function, m::Int, n::Int) = SimpleRecurrent(
    f=f,
    w=orthonormal(f, m, n),
    u=orthonormal(f, m, m),
    b=zeros(m),
    h0=zeros(m),
)

SimpleRecurrent(m::Int, n::Int) = SimpleRecurrent(tanh, m, n)

@component Base.step(p::SimpleRecurrent, x::Variable, htm1::Variable) = p.f(plus(linear(p.w, x), linear(p.u, htm1), p.b))

@component Base.step(p::SimpleRecurrent, x::Variable) = step(p, x, p.h0)

@component function unfold{T<:Variable}(p::SimpleRecurrent, x::Vector{T})
    @similar_variable_type H T
    h = Array(H, length(x))
    h[1] = step(p, x[1])
    for t = 2:length(x)
        h[t] = step(p, x[t], h[t-1])
    end
    return h
end
