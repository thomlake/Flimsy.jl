immutable Recurrent{T,M,N} <: Component
    f::Function
    w::Var{T,M,N}
    u::Var{T,M,M}
    b::Var{T,M,1}
    h0::Var{T,M,1}
end

Recurrent(f::Function, m::Int, n::Int) = Recurrent(f, Orthonormal(m, n), Orthonormal(m, m), Zeros(m), Zeros(m))
Recurrent(m::Int, n::Int) = Recurrent(tanh, m, n)

@Nimble.component Base.step(theta::Recurrent, x::Var) = theta.f(affine(theta.w, x, theta.h0))

@Nimble.component Base.step(theta::Recurrent, x::Var, htm1) = theta.f(sum(linear(theta.w, x), linear(theta.u, htm1), theta.b))

@Nimble.component function unfold(theta::Recurrent, x::Vector)
    h = Array(Var, length(x))
    h[1] = step(theta, x[1])
    for t = 2:length(x)
        h[t] = step(theta, x[t], h[t-1])
    end
    return h
end
