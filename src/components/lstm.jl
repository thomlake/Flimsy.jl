
immutable Lstm <: RecurrentComponent2
    wi::Variable; wf::Variable; wc::Variable; wo::Variable;
    ui::Variable; uf::Variable; uc::Variable; uo::Variable;
    bi::Variable; bf::Variable; bc::Variable; bo::Variable;
    h_init::Variable; c_init::Variable;
    function Lstm(wi, wf, wc, wo, ui, uf, uc, uo, bi, bf, bc, bo, h_init, c_init)
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
        
        size(h_init) == (m, 1) || error("Bad size(h_init) == $(size(h_init)) != ($m, 1)")
        size(c_init) == (m, 1) || error("Bad size(c_init) == $(size(c_init)) != ($m, 1)")

        return new(wi, wf, wc, wo, ui, uf, uc, uo, bi, bf, bc, bo, h_init, c_init)
    end
end

initial_state(scope::Scope, params::Lstm) = (params.h_init, params.c_init)

function Base.step(scope::Scope, params::Lstm, x, state)
    htm1, ctm1 = state
    @with scope begin
        i = sigmoid(plus(linear(params.wi, x), linear(params.ui, htm1), params.bi))
        f = sigmoid(plus(linear(params.wf, x), linear(params.uf, htm1), params.bf))
        o = sigmoid(plus(linear(params.wo, x), linear(params.uo, htm1), params.bo))
        c = tanh(plus(linear(params.wc, x), linear(params.uc, htm1), params.bc))
        ct = plus(mult(i, c), mult(f, ctm1))
        ht = mult(o, tanh(ct))
    end
    return (ht, ct)
end

Base.step(scope::Scope, params::Lstm, x) = @with scope step(params, x, initial_state(params))

function unfold(scope::Scope, params::Lstm, x::Vector)
    @with scope begin
        h = Sequence(length(x))
        h[1], c = step(params, x[1])
        for t = 2:length(x)
            h[t], c = step(params, x[t], (h[t-1], c))
        end
    end
    return h
end

"""
Lstm Component with normalized hidden unit gradients.
By default gradients are normalized to 1/timesteps.
"""
immutable LstmGradNorm <: RecurrentComponent2
    wi::Variable; wf::Variable; wc::Variable; wo::Variable;
    ui::Variable; uf::Variable; uc::Variable; uo::Variable;
    bi::Variable; bf::Variable; bc::Variable; bo::Variable;
    h_init::Variable; c_init::Variable;
    function LstmGradNorm(wi, wf, wc, wo, ui, uf, uc, uo, bi, bf, bc, bo, h_init, c_init)
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
        
        size(h_init) == (m, 1) || error("Bad size(h_init) == $(size(h_init)) != ($m, 1)")
        size(c_init) == (m, 1) || error("Bad size(c_init) == $(size(c_init)) != ($m, 1)")

        return new(wi, wf, wc, wo, ui, uf, uc, uo, bi, bf, bc, bo, h_init, c_init)
    end
end

# LstmGradNorm{V<:Variable}(wi::V, wf::V, wc::V, wo::V, ui::V, uf::V, uc::V, uo::V, bi::V, bf::V, bc::V, bo::V, h0::V, c0::V) = 
#     LstmGradNorm{V}(wi, wf, wc, wo, ui, uf, uc, uo, bi, bf, bc, bo, h0, c0)

initial_state(scope::Scope, params::LstmGradNorm) = (params.h_init, params.c_init)

function Base.step(scope::Scope, params::LstmGradNorm, x, state, gn::AbstractFloat=1.0)
    htm1, ctm1 = state
    @with scope begin
        i = sigmoid(plus(linear(params.wi, x), linear(params.ui, htm1), params.bi))
        f = sigmoid(plus(linear(params.wf, x), linear(params.uf, htm1), params.bf))
        o = sigmoid(plus(linear(params.wo, x), linear(params.uo, htm1), params.bo))
        c = tanh(plus(linear(params.wc, x), linear(params.uc, htm1), params.bc))
        ct = plus(mult(i, c), mult(f, ctm1))
        gradnorm(ct, gn)
        ht = mult(o, tanh(ct))
    end
    return (ht, ct)
end

Base.step(scope::Scope, params::LstmGradNorm, x, gn::AbstractFloat=1.0) = @with scope step(params, x, initial_state(params), gn)

function unfold(scope::Scope, params::LstmGradNorm, x::Vector, gn::AbstractFloat=inv(length(x)))
    @with scope begin
        h = Sequence(length(x))
        h[1], c = step(params, x[1], gn)
        for t = 2:length(x)
            h[t], c = step(params, x[t], (h[t-1], c), gn)
        end
    end
    return h
end

"""
Convenience Constructor
"""
function Lstm(m::Int, n::Int; normed::Bool=false)
    # dist = Normal(0, 0.1)
    # wi, wf, wc, wo = rand(dist, m, n), rand(dist, m, n), rand(dist, m, n), rand(dist, m, n)
    wi, wf, wc, wo = orthonormal(m, n), orthonormal(m, n), orthonormal(m, n), orthonormal(m, n)
    ui, uf, uc, uo = orthonormal(m, m), orthonormal(m, m), orthonormal(m, m), orthonormal(m, m)
    bi, bf, bc, bo = zeros(m, 1), ones(m, 1), zeros(m, 1), zeros(m, 1)
    h_init, c_init = zeros(m, 1), zeros(m, 1)
    if normed
        return LstmGradNorm(
            wi=wi, wf=wf, wc=wc, wo=wo, 
            ui=ui, uf=uf, uc=uc, uo=uo, 
            bi=bi, bf=bf, bc=bc, bo=bo,
            h_init=h_init, 
            c_init=c_init,
        )
    else
        return Lstm(
            wi=wi, wf=wf, wc=wc, wo=wo, 
            ui=ui, uf=uf, uc=uc, uo=uo, 
            bi=bi, bf=bf, bc=bc, bo=bo,
            h_init=h_init, 
            c_init=c_init,
        )
    end
end
