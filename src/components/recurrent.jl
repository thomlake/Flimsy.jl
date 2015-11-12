immutable Recurrent{T,M,N} <: Component
    f::Function
    w::Variable{T,M,N}
    u::Variable{T,M,M}
    b::Variable{T,M,1}
    h0::Variable{T,M,1}
end

Recurrent(f::Function, m::Int, n::Int) = Recurrent(f, Orthonormal(m, n), Orthonormal(m, m), Zeros(m), Zeros(m))
Recurrent(m::Int, n::Int) = Recurrent(tanh, m, n)

@flimsy Base.step(theta::Recurrent, x::Variable) = theta.f(affine(theta.w, x, theta.h0))

@flimsy Base.step(theta::Recurrent, x::Variable, htm1) = theta.f(sum(linear(theta.w, x), linear(theta.u, htm1), theta.b))

@flimsy function unfold(theta::Recurrent, x::Vector)
    h = Array(Variable, length(x))
    h[1] = step(theta, x[1])
    for t = 2:length(x)
        h[t] = step(theta, x[t], h[t-1])
    end
    return h
end
