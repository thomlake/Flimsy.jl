
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

@comp initial_state(params::Lstm) = (params.h0, params.c0)

@comp function Base.step(params::Lstm, x, state)
    htm1, ctm1 = state
    i = sigmoid(plus(linear(params.wi, x), linear(params.ui, htm1), params.bi))
    f = sigmoid(plus(linear(params.wf, x), linear(params.uf, htm1), params.bf))
    o = sigmoid(plus(linear(params.wo, x), linear(params.uo, htm1), params.bo))
    c = tanh(plus(linear(params.wc, x), linear(params.uc, htm1), params.bc))
    ct = plus(mult(i, c), mult(f, ctm1))
    return (mult(o, tanh(ct)), ct)
end

@comp Base.step(params::Lstm, x) = step(params, x, initial_state(params))

@comp function unfold(params::Lstm, x::Vector)
    h = Sequence(length(x))
    h[1], c = step(params, x[1])
    for t = 2:length(x)
        h[t], c = step(params, x[t], (h[t-1], c))
    end
    return h
end

"""
Lstm Component with normalized hidden unit gradients.
By default gradients are normalized to 1/timesteps.
"""
immutable LstmGradNorm{V<:Variable} <: RecurrentComponent2{V}
    wi::V; wf::V; wc::V; wo::V;
    ui::V; uf::V; uc::V; uo::V;
    bi::V; bf::V; bc::V; bo::V;
    h0::V; c0::V
    function LstmGradNorm(wi::V, wf::V, wc::V, wo::V, ui::V, uf::V, uc::V, uo::V, bi::V, bf::V, bc::V, bo::V, h0::V, c0::V)
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

LstmGradNorm{V<:Variable}(wi::V, wf::V, wc::V, wo::V, ui::V, uf::V, uc::V, uo::V, bi::V, bf::V, bc::V, bo::V, h0::V, c0::V) = 
    LstmGradNorm{V}(wi, wf, wc, wo, ui, uf, uc, uo, bi, bf, bc, bo, h0, c0)

@comp initial_state(params::LstmGradNorm) = (params.h0, params.c0)

@comp function Base.step(params::LstmGradNorm, x, state, gn::AbstractFloat=1.0)
    htm1, ctm1 = state
    i = sigmoid(plus(linear(params.wi, x), linear(params.ui, htm1), params.bi))
    f = sigmoid(plus(linear(params.wf, x), linear(params.uf, htm1), params.bf))
    o = sigmoid(plus(linear(params.wo, x), linear(params.uo, htm1), params.bo))
    c = tanh(plus(linear(params.wc, x), linear(params.uc, htm1), params.bc))
    ct = plus(mult(i, c), mult(f, ctm1))
    gradnorm(ct, gn)
    h = tanh(ct)
    return (mult(o, h), ct)
end

@comp Base.step(params::LstmGradNorm, x, gn::AbstractFloat=1.0) = step(params, x, initial_state(params), gn)

@comp function unfold(params::LstmGradNorm, x::Vector, gn::AbstractFloat=inv(length(x)))
    h = Sequence(length(x))
    h[1], c = step(params, x[1], gn)
    for t = 2:length(x)
        h[t], c = step(params, x[t], (h[t-1], c), gn)
    end
    return h
end

"""
Convenience Constructor
"""
function Lstm(m::Int, n::Int; normed::Bool=false)
    dist = Normal(0, 0.01)
    wi, wf, wc, wo = rand(dist, m, n), rand(dist, m, n), rand(dist, m, n), rand(dist, m, n)
    ui, uf, uc, uo = orthonormal(m, m), orthonormal(m, m), orthonormal(m, m), orthonormal(m, m)
    bi, bf, bc, bo = zeros(m, 1), ones(m, 1), zeros(m, 1), zeros(m, 1)
    h0, c0 = zeros(m, 1), zeros(m, 1)
    if normed
        return LstmGradNorm(
            wi=wi, wf=wf, wc=wc, wo=wo, 
            ui=ui, uf=uf, uc=uc, uo=uo, 
            bi=bi, bf=bf, bc=bc, bo=bo,
            h0=h0, c0=c0
        )
    else
        return Lstm(
            wi=wi, wf=wf, wc=wc, wo=wo, 
            ui=ui, uf=uf, uc=uc, uo=uo, 
            bi=bi, bf=bf, bc=bc, bo=bo,
            h0=h0, c0=c0
        )
    end
end