immutable SimpleRecurrent{T,M,N} <: RecurrentComponent{T,M,N}
    f::Function
    w::Variable{T,M,N}
    u::Variable{T,M,M}
    b::Variable{T,M,1}
    h0::Variable{T,M,1}
end

SimpleRecurrent(f::Function, m::Int, n::Int) = SimpleRecurrent(f, Orthonormal(m, n), Orthonormal(m, m), Zeros(m), Zeros(m))
SimpleRecurrent(m::Int, n::Int) = SimpleRecurrent(tanh, m, n)

@flimsy Base.step(theta::SimpleRecurrent, x::Variable) = theta.f(affine(theta.w, x, theta.h0))

@flimsy Base.step(theta::SimpleRecurrent, x::Variable, htm1) = theta.f(sum(linear(theta.w, x), linear(theta.u, htm1), theta.b))

@flimsy function unfold(theta::SimpleRecurrent, x::Vector)
    h = Array(Variable, length(x))
    h[1] = step(theta, x[1])
    for t = 2:length(x)
        h[t] = step(theta, x[t], h[t-1])
    end
    return h
end
