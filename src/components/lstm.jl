
immutable Lstm{V<:Variable} <: RecurrentComponent2{V}
    wi::V; wf::V; wc::V; wo::V;
    ui::V; uf::V; uc::V; uo::V;
    bi::V; bf::V; bc::V; bo::V;
    h0::V; c0::V
    function Lstm(wi::V, wf::V, wc::V, wo::V, ui::V, uf::V, uc::V, uo::V, bi::V, bf::V, bc::V, bo::V, h0::V, c0::V)
        m, n = size(wi)
        size(wf) == (m, n) || error("Bad size(wf) == $(size(wf)) != ($m, $n)")
        size(wc) == (m, n) || error("Bad size(wc) == $(size(wc)) != ($m, $n)")
        size(wo) == (m, n) || error("Bad size(wo) == $(size(wo)) != ($m, $n)")
        
        size(ui) == (m, m) || error("Bad size(ui) == $(size(ui)) != ($m, $m)")
        size(uf) == (m, m) || error("Bad size(uf) == $(size(uf)) != ($m, $m)")
        size(uc) == (m, m) || error("Bad size(uc) == $(size(uc)) != ($m, $m)")
        size(uo) == (m, m) || error("Bad size(uo) == $(size(uo)) != ($m, $m)")

        size(bi) == (m, 1) || error("Bad size(bi) == $(size(bi)) != ($m, 1)")
        size(bf) == (m, 1) || error("Bad size(bf) == $(size(bf)) != ($m, 1)")
        size(bc) == (m, 1) || error("Bad size(bc) == $(size(bc)) != ($m, 1)")
        size(bo) == (m, 1) || error("Bad size(bo) == $(size(bo)) != ($m, 1)")
        
        size(h0) == (m, 1) || error("Bad size(h0) == $(size(h0)) != ($m, 1)")
        size(c0) == (m, 1) || error("Bad size(c0) == $(size(c0)) != ($m, 1)")

        return new(wi, wf, wc, wo, ui, uf, uc, uo, bi, bf, bc, bo, h0, c0)
    end
end
Lstm{V<:Variable}(wi::V, wf::V, wc::V, wo::V, ui::V, uf::V, uc::V, uo::V, bi::V, bf::V, bc::V, bo::V, h0::V, c0::V) = 
    Lstm{V}(wi, wf, wc, wo, ui, uf, uc, uo, bi, bf, bc, bo, h0, c0)

Lstm(m::Int, n::Int) = Lstm(
    wi=orthonormal(m, n), wf=orthonormal(m, n), wc=orthonormal(m, n), wo=orthonormal(m, n),
    ui=orthonormal(m, m), uf=orthonormal(m, m), uc=orthonormal(m, m), uo=orthonormal(m, m),
    bi=zeros(m, 1), bf=zeros(m, 1), bc=zeros(m, 1), bo=zeros(m, 1),
    h0=zeros(m, 1), c0=zeros(m, 1),
)

@component function Base.step(params::Lstm, x, state)
    htm1, ctm1 = state
    i = sigmoid(plus(linear(params.wi, x), linear(params.ui, htm1), params.bi))
    f = sigmoid(plus(linear(params.wf, x), linear(params.uf, htm1), params.bf))
    o = sigmoid(plus(linear(params.wo, x), linear(params.uo, htm1), params.bo))
    c = tanh(plus(linear(params.wc, x), linear(params.uc, htm1), params.bc))
    ct = plus(mult(i, c), mult(f, ctm1))
    return (mult(o, tanh(ct)), ct)
end

@component Base.step(params::Lstm, x) = step(params, x, (params.h0, params.c0))

@component function unfold(params::Lstm, x::Vector)
    h1, c1 = step(params, x)
    h, c = Array(typeof(h1), length(x)), Array(typeof(c1), length(x))
    h[1], c[1] = h1, c1
    for t = 2:length(x)
        h[t], c[t] = step(params, x[t], (h[t-1], c[t-1]))
    end
    return h
end
