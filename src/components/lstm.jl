
immutable LSTM{T,M,N} <: RecurrentComponent{T,M,N}
    wi::Variable{T,M,N}; wf::Variable{T,M,N}; wc::Variable{T,M,N}; wo::Variable{T,M,N};
    ui::Variable{T,M,M}; uf::Variable{T,M,M}; uc::Variable{T,M,M}; uo::Variable{T,M,M}; vo::Variable{T,M,M};
    bi::Variable{T,M,1}; bf::Variable{T,M,1}; bc::Variable{T,M,1}; bo::Variable{T,M,1};
    h0::Variable{T,M,1}; c0::Variable{T,M,1}
end

LSTM(m::Int, n::Int) = LSTM(
    Orthonormal(m, n), Orthonormal(m, n), Orthonormal(m, n), Orthonormal(m, n),
    Orthonormal(m, m), Orthonormal(m, m), Orthonormal(m, m), Orthonormal(m, m), Orthonormal(m, m),
    Zeros(m), Zeros(m), Zeros(m), Zeros(m),
    Zeros(m), Zeros(m),
)

@flimsy function Base.step(theta::LSTM, x::Variable)
    i = sigmoid(sum(linear(theta.wi, x), linear(theta.ui, theta.h0), theta.bi))
    f = sigmoid(sum(linear(theta.wf, x), linear(theta.uf, theta.h0), theta.bf))
    cnew = tanh(sum(linear(theta.wc, x), linear(theta.uc, theta.h0), theta.bc))
    c = sum(prod(i, cnew), prod(f, theta.c0))
    o = sigmoid(sum(linear(theta.wo, x), linear(theta.uo, theta.h0), linear(theta.vo, c), theta.bo))
    return (prod(o, tanh(c)), c)
end

@flimsy function Base.step{V1<:Variable,V2<:Variable}(theta::LSTM, x::Variable, state::Tuple{V1,V2})
    htm1, ctm1 = state
    i = sigmoid(sum(linear(theta.wi, x), linear(theta.ui, htm1), theta.bi))
    f = sigmoid(sum(linear(theta.wf, x), linear(theta.uf, htm1), theta.bf))
    cnew = tanh(sum(linear(theta.wc, x), linear(theta.uc, htm1), theta.bc))
    c = sum(prod(i, cnew), prod(f, ctm1))
    o = sigmoid(sum(linear(theta.wo, x), linear(theta.uo, theta.h0), linear(theta.vo, c), theta.bo))
    return (prod(o, tanh(c)), c)
end

@flimsy function unfold(theta::LSTM, x::Vector)
    h = Array(Variable, length(x))
    c = Array(Variable, length(x))
    h[1], c[1] = step(theta, x[1])
    for t = 2:length(x)
        h[t], c[t] = step(theta, x[t], (h[t-1], c[t-1]))
    end
    return h
end
