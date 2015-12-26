
immutable ResidualRecurrent{T,M,N,H} <: RecurrentComponent{T,M,N}
    w::Variable{T,H,N} 
    u::Variable{T,H,M}
    v::Variable{T,M,H}
    bh::Variable{T,H,1}
    br::Variable{T,M,1}
    h0::Variable{T,M,1}
end

ResidualRecurrent(m::Int, h::Int, n::Int) = ResidualRecurrent(
    Orthonormal(h, n), 
    Orthonormal(h, m), 
    Orthonormal(m, h),
    Zeros(h), 
    Zeros(m), 
    Zeros(m),
)

ResidualRecurrent(m::Int, n::Int) = ResidualRecurrent(
    Orthonormal(m, n), 
    Orthonormal(m, m), 
    Orthonormal(m, m),
    Zeros(m), 
    Zeros(m), 
    Zeros(m),
)

@flimsy function Base.step(theta::ResidualRecurrent, x::Variable, stm1)
    h = relu(sum(linear(theta.w, x), linear(theta.u, stm1), theta.bh))
    r = tanh(affine(theta.v, h, theta.br))
    return sum(stm1, r)
end

@flimsy Base.step(theta::ResidualRecurrent, x::Variable) = step(theta, x, theta.h0)

@flimsy function unfold(theta::ResidualRecurrent, x::Vector)
    s = Array(Variable, length(x))
    s[1] = step(theta, x[1])
    for t = 2:length(x)
        s[t] = step(theta, x[t], s[t-1])
    end
    return s
end
