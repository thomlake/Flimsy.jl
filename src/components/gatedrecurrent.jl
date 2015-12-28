
immutable GatedRecurrent{T,M,N} <: RecurrentComponent{T,M,N}
    wr::Variable{T,M,N}; wz::Variable{T,M,N}; wc::Variable{T,M,N};
    ur::Variable{T,M,M}; uz::Variable{T,M,M}; uc::Variable{T,M,M};
    br::Variable{T,M,1}; bz::Variable{T,M,1}; bc::Variable{T,M,1};
    h0::Variable{T,M,1}
end


GatedRecurrent(m::Int, n::Int) = GatedRecurrent(
    wr=orthonormal(m, n), wz=orthonormal(m, n), wc=orthonormal(m, n),
    ur=orthonormal(m, m), uz=orthonormal(m, m), uc=orthonormal(m, m),
    br=zeros(m), bz=zeros(m), bc=zeros(m), h0=zeros(m),
)

@flimsy function Base.step(theta::GatedRecurrent, x::Variable, htm1)
    r = sigmoid(sum(linear(theta.wr, x), linear(theta.ur, htm1), theta.br))
    z = sigmoid(sum(linear(theta.wz, x), linear(theta.uz, htm1), theta.bz))
    c = tanh(sum(linear(theta.wc, x), prod(r, linear(theta.uc, htm1)), theta.bc))
    return sum(prod(z, htm1), prod(minus(1.0, z), c))
end

@flimsy Base.step(theta::GatedRecurrent, x::Variable) = step(theta, x, theta.h0)

@flimsy function unfold(theta::GatedRecurrent, x::Vector)
    h = Array(Variable, length(x))
    h[1] = step(theta, x[1])
    for t = 2:length(x)
        h[t] = step(theta, x[t], h[t-1])
    end
    return h
end
