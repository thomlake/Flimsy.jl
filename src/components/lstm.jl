immutable LSTM{T,M,N} <: Component
    wi::Var{T,M,N}; wf::Var{T,M,N}; wc::Var{T,M,N}; wo::Var{T,M,N};
    ui::Var{T,M,M}; uf::Var{T,M,M}; uc::Var{T,M,M}; uo::Var{T,M,M}; vo::Var{T,M,M};
    bi::Var{T,M,1}; bf::Var{T,M,1}; bc::Var{T,M,1}; bo::Var{T,M,1};
    h0::Var{T,M,1}; c0::Var{T,M,1}
end

LSTM(m::Int, n::Int) = LSTM(
    Orthonormal(m, n), Orthonormal(m, n), Orthonormal(m, n), Orthonormal(m, n),
    Orthonormal(m, m), Orthonormal(m, m), Orthonormal(m, m), Orthonormal(m, m), Orthonormal(m, m),
    Zeros(m), Zeros(m), Zeros(m), Zeros(m),
    Zeros(m), Zeros(m),
)

@Nimble.component function Base.step(theta::LSTM, x::Var)
    i = sigmoid(sum(linear(theta.wi, x), linear(theta.ui, theta.h0), theta.bi))
    f = sigmoid(sum(linear(theta.wf, x), linear(theta.uf, theta.h0), theta.bf))
    cnew = tanh(sum(linear(theta.wc, x), linear(theta.uc, theta.h0), theta.bc))
    c = sum(prod(i, cnew), prod(f, theta.c0))
    o = sigmoid(sum(linear(theta.wo, x), linear(theta.uo, theta.h0), linear(theta.vo, c), theta.bo))
    return prod(o, tanh(c)), c
end

@Nimble.component function Base.step(theta::LSTM, x::Var, htm1, ctm1)
    i = sigmoid(sum(linear(theta.wi, x), linear(theta.ui, htm1), theta.bi))
    f = sigmoid(sum(linear(theta.wf, x), linear(theta.uf, htm1), theta.bf))
    cnew = tanh(sum(linear(theta.wc, x), linear(theta.uc, htm1), theta.bc))
    c = sum(prod(i, cnew), prod(f, ctm1))
    o = sigmoid(sum(linear(theta.wo, x), linear(theta.uo, theta.h0), linear(theta.vo, c), theta.bo))
    return prod(o, tanh(c)), c
end

@Nimble.component function unfold(theta::LSTM, x::Vector)
    h = Array(Var, length(x))
    c = Array(Var, length(x))
    h[1], c[1] = step(theta, x[1])
    for t = 2:length(x)
        h[t], c[t] = step(theta, x[t], h[t-1], c[t-1])
    end
    return h
end
