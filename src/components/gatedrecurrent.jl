immutable GatedRecurrent{T,M,N} <: RecurrentComponent{T,M,N}
    wr::Variable{T,M,N}; wz::Variable{T,M,N}; wc::Variable{T,M,N};
    ur::Variable{T,M,M}; uz::Variable{T,M,M}; uc::Variable{T,M,M};
    br::Variable{T,M,1}; bz::Variable{T,M,1}; bc::Variable{T,M,1};
    h0::Variable{T,M,1}
end

GatedRecurrent(m::Int, n::Int) = GatedRecurrent(
    Orthonormal(m, n), Orthonormal(m, n), Orthonormal(m, n),
    Orthonormal(m, m), Orthonormal(m, m), Orthonormal(m, m),
    Zeros(m), Zeros(m), Zeros(m),
    Zeros(m)
)

@flimsy function Base.step(theta::GatedRecurrent, x::Variable)
    r = sigmoid(affine(theta.wr, x, theta.h0))
    z = sigmoid(affine(theta.wz, x, theta.h0))
    c = tanh(affine(theta.wc, x, prod(r, theta.h0)))
    return sum(prod(z, theta.h0), prod(minus(1.0, z), c))
end

@flimsy function Base.step(theta::GatedRecurrent, x::Variable, htm1)
    r = sigmoid(sum(linear(theta.wr, x), linear(theta.ur, htm1), theta.br))
    z = sigmoid(sum(linear(theta.wz, x), linear(theta.uz, htm1), theta.bz))
    c = tanh(sum(linear(theta.wc, x), prod(r, linear(theta.uc, htm1)), theta.bc))
    return sum(prod(z, htm1), prod(minus(1.0, z), c))
end

@flimsy function unfold(theta::GatedRecurrent, x::Vector)
    h = Array(Variable, length(x))
    h[1] = step(theta, x[1])
    for t = 2:length(x)
        h[t] = step(theta, x[t], h[t-1])
    end
    return h
end
