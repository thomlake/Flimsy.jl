immutable GatedRecurrent{T,M,N} <: Component
    wr::Var{T,M,N}; wz::Var{T,M,N}; wc::Var{T,M,N};
    ur::Var{T,M,M}; uz::Var{T,M,M}; uc::Var{T,M,M};
    br::Var{T,M,1}; bz::Var{T,M,1}; bc::Var{T,M,1};
    h0::Var{T,M,1}
end

GatedRecurrent(m::Int, n::Int) = GatedRecurrent(
    Orthonormal(m, n), Orthonormal(m, n), Orthonormal(m, n),
    Orthonormal(m, m), Orthonormal(m, m), Orthonormal(m, m),
    Zeros(m), Zeros(m), Zeros(m),
    Zeros(m)
)

@Nimble.component function Base.step(theta::GatedRecurrent, x::Var)
    r = sigmoid(affine(theta.wr, x, theta.h0))
    z = sigmoid(affine(theta.wz, x, theta.h0))
    c = tanh(affine(theta.wc, x, prod(r, theta.h0)))
    return sum(prod(z, theta.h0), prod(minus(1.0, z), c))
end

@Nimble.component function Base.step(theta::GatedRecurrent, x::Var, htm1)
    r = sigmoid(sum(linear(theta.wr, x), linear(theta.ur, htm1), theta.br))
    z = sigmoid(sum(linear(theta.wz, x), linear(theta.uz, htm1), theta.bz))
    c = tanh(sum(linear(theta.wc, x), prod(r, linear(theta.uc, htm1)), theta.bc))
    return sum(prod(z, htm1), prod(minus(1.0, z), c))
end

@Nimble.component function unfold(theta::GatedRecurrent, x::Vector)
    h = Array(Var, length(x))
    h[1] = step(theta, x[1])
    for t = 2:length(x)
        h[t] = step(theta, x[t], h[t-1])
    end
    return h
end
