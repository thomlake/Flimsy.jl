
immutable LSTM{V<:Variable} <: RecurrentComponent{V}
    wi::V; wf::V; wc::V; wo::V;
    ui::V; uf::V; uc::V; uo::V; vo::V;
    bi::V; bf::V; bc::V; bo::V;
    h0::V; c0::V
    function LSTM(wi, wf, wc, wo, ui, uf, uc, uo, vo, bi, bf, bc, bo, h0, c0)
        m, n = size(wi)
        size(wf) == (m, n) || error("Bad size(wf) == $(size(wf)) != ($m, $n)")
        size(wc) == (m, n) || error("Bad size(wc) == $(size(wc)) != ($m, $n)")
        size(wo) == (m, n) || error("Bad size(wo) == $(size(wo)) != ($m, $n)")
        
        size(ui) == (m, m) || error("Bad size(ui) == $(size(ui)) != ($m, $m)")
        size(uf) == (m, m) || error("Bad size(uf) == $(size(uf)) != ($m, $m)")
        size(uc) == (m, m) || error("Bad size(uc) == $(size(uc)) != ($m, $m)")
        size(uo) == (m, m) || error("Bad size(uo) == $(size(uo)) != ($m, $m)")
        size(vo) == (m, m) || error("Bad size(vo) == $(size(vo)) != ($m, $m)")

        size(bi) == (m, 1) || error("Bad size(bi) == $(size(bi)) != ($m, 1)")
        size(bf) == (m, 1) || error("Bad size(bf) == $(size(bf)) != ($m, 1)")
        size(bc) == (m, 1) || error("Bad size(bc) == $(size(bc)) != ($m, 1)")
        size(bo) == (m, 1) || error("Bad size(bo) == $(size(bo)) != ($m, 1)")
        
        size(h0) == (m, 1) || error("Bad size(h0) == $(size(h0)) != ($m, 1)")
        size(c0) == (m, 1) || error("Bad size(c0) == $(size(c0)) != ($m, 1)")

        return new(wi, wf, wc, wo, ui, uf, uc, uo, vo, bi, bf, bc, bo, h0, c0)
    end
end

LSTM(m::Int, n::Int) = LSTM(
    wi=orthonormal(m, n), wf=orthonormal(m, n), wc=orthonormal(m, n), wo=orthonormal(m, n),
    ui=orthonormal(m, m), uf=orthonormal(m, m), uc=orthonormal(m, m), uo=orthonormal(m, m), vo=orthonormal(m, m),
    bi=zeros(m), bf=zeros(m), bc=zeros(m), bo=zeros(m),
    h0=zeros(m), c0=zeros(m),
)

@component function Base.step(params::LSTM, x::Variable, state::Tuple{Variable,Variable})
    htm1, ctm1 = state
    i = sigmoid(plus(linear(params.wi, x), linear(params.ui, htm1), params.bi))
    f = sigmoid(plus(linear(params.wf, x), linear(params.uf, htm1), params.bf))
    cnew = tanh(plus(linear(params.wc, x), linear(params.uc, htm1), params.bc))
    c = plus(mult(i, cnew), mult(f, ctm1))
    o = sigmoid(plus(linear(params.wo, x), linear(params.uo, params.h0), linear(params.vo, c), params.bo))
    return (mult(o, tanh(c)), c)
end

@component Base.step(params::LSTM, x::Variable) = step(params, x, (params.h0, params.c0))

@component function unfold{T<:Variable}(params::LSTM, x::Vector{T})
    @similar_variable_type H T
    h = Array(H, length(x))
    c = Array(H, length(x))
    h[1], c[1] = step(params, x[1])
    for t = 2:length(x)
        h[t], c[t] = step(params, x[t], (h[t-1], c[t-1]))
    end
    return h
end
