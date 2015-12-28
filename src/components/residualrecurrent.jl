
immutable ResidualRecurrent{T,M,N,H} <: RecurrentComponent{T,M,N}
    w::Variable{T,H,N} 
    u::Variable{T,H,M}
    v::Variable{T,M,H}
    bh::Variable{T,H,1}
    br::Variable{T,M,1}
    h0::Variable{T,M,1}
end

ResidualRecurrent(m::Int, h::Int, n::Int) = ResidualRecurrent(
    w=rand(Normal(0, 0.01), h, n), 
    u=rand(Normal(0, 0.01), h, m), 
    v=rand(Normal(0, 0.01), m, h),
    bh=zeros(h), 
    br=zeros(m), 
    h0=zeros(m),
)

ResidualRecurrent(m::Int, n::Int) = ResidualRecurrent(m, max(10, round(Int, m / 2)), n)

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
