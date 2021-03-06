
immutable GatedRecurrent{V<:Variable} <: RecurrentComponent1{V}
    wr::V; wz::V; wc::V;
    ur::V; uz::V; uc::V;
    br::V; bz::V; bc::V;
    h0::V
    function GatedRecurrent(wr::V, wz::V, wc::V, ur::V, uz::V, uc::V, br::V, bz::V, bc::V, h0::V)
        m, n = size(wr)
        size(wz) == (m, n) || error("Bad size(wz) == $(size(wz)) != ($m, $n)")
        size(wc) == (m, n) || error("Bad size(wc) == $(size(wc)) != ($m, $n)")
        size(ur) == (m, m) || error("Bad size(ur) == $(size(ur)) != ($m, $m)")
        size(uz) == (m, m) || error("Bad size(uz) == $(size(uz)) != ($m, $m)")
        size(uc) == (m, m) || error("Bad size(uc) == $(size(uc)) != ($m, $m)")
        size(br) == (m, 1) || error("Bad size(br) == $(size(br)) != ($m, 1)")
        size(bz) == (m, 1) || error("Bad size(bz) == $(size(bz)) != ($m, 1)")
        size(bc) == (m, 1) || error("Bad size(bc) == $(size(bc)) != ($m, 1)")
        size(h0) == (m, 1) || error("Bad size(h0) == $(size(h0)) != ($m, 1)")
        return new(wr, wz, wc, ur, uz, uc, br, bz, bc, h0)
    end
end

GatedRecurrent{V<:Variable}(wr::V, wz::V, wc::V, ur::V, uz::V, uc::V, br::V, bz::V, bc::V, h0::V) = 
    GatedRecurrent{V}(wr, wz, wc, ur, uz, uc, br, bz, bc, h0)

@comp initial_state(params::GatedRecurrent) = params.h0

@comp function Base.step(params::GatedRecurrent, x, htm1)
    r = sigmoid(plus(linear(params.wr, x), linear(params.ur, htm1), params.br))
    z = sigmoid(plus(linear(params.wz, x), linear(params.uz, htm1), params.bz))
    c = tanh(plus(linear(params.wc, x), mult(r, linear(params.uc, htm1)), params.bc))
    return plus(mult(z, htm1), mult(minus(1.0, z), c))
end

@comp Base.step(params::GatedRecurrent, x) = step(params, x, initial_state(params))

@comp function unfold(params::GatedRecurrent, x::Vector, h0::Variable) 
    h = Sequence(length(x))
    h[1] = step(params, x[1], h0)
    for t = 2:length(x)
        h[t] = step(params, x[t], h[t-1])
    end
    return h
end

@comp unfold(params::GatedRecurrent, x::Vector) = unfold(params, x, initial_state(params))

"""
GatedRecurrent Component with normalized hidden unit gradients.
By default gradients are normalized to 1/timesteps.
"""
immutable GatedRecurrentGradNorm{V<:Variable} <: RecurrentComponent1{V}
    wr::V; wz::V; wc::V;
    ur::V; uz::V; uc::V;
    br::V; bz::V; bc::V;
    h0::V
    function GatedRecurrentGradNorm(wr::V, wz::V, wc::V, ur::V, uz::V, uc::V, br::V, bz::V, bc::V, h0::V)
        m, n = size(wr)
        size(wz) == (m, n) || error("Bad size(wz) == $(size(wz)) != ($m, $n)")
        size(wc) == (m, n) || error("Bad size(wc) == $(size(wc)) != ($m, $n)")
        size(ur) == (m, m) || error("Bad size(ur) == $(size(ur)) != ($m, $m)")
        size(uz) == (m, m) || error("Bad size(uz) == $(size(uz)) != ($m, $m)")
        size(uc) == (m, m) || error("Bad size(uc) == $(size(uc)) != ($m, $m)")
        size(br) == (m, 1) || error("Bad size(br) == $(size(br)) != ($m, 1)")
        size(bz) == (m, 1) || error("Bad size(bz) == $(size(bz)) != ($m, 1)")
        size(bc) == (m, 1) || error("Bad size(bc) == $(size(bc)) != ($m, 1)")
        size(h0) == (m, 1) || error("Bad size(h0) == $(size(h0)) != ($m, 1)")
        return new(wr, wz, wc, ur, uz, uc, br, bz, bc, h0)
    end
end

GatedRecurrentGradNorm{V<:Variable}(wr::V, wz::V, wc::V, ur::V, uz::V, uc::V, br::V, bz::V, bc::V, h0::V) = 
    GatedRecurrentGradNorm{V}(wr, wz, wc, ur, uz, uc, br, bz, bc, h0)

@comp initial_state(params::GatedRecurrentGradNorm) = params.h0

@comp function Base.step(params::GatedRecurrentGradNorm, x, htm1, gn::AbstractFloat=1.0)
    r = sigmoid(plus(linear(params.wr, x), linear(params.ur, htm1), params.br))
    z = sigmoid(plus(linear(params.wz, x), linear(params.uz, htm1), params.bz))
    c_pre = plus(linear(params.wc, x), mult(r, linear(params.uc, htm1)), params.bc)
    gradnorm(c_pre, gn)
    c = tanh(c_pre)
    return plus(mult(z, htm1), mult(minus(1.0, z), c))
end

@comp Base.step(params::GatedRecurrentGradNorm, x, gn::AbstractFloat=1.0) = step(params, x, initial_state(params), gn)

@comp function unfold(params::GatedRecurrentGradNorm, x::Vector, h0::Variable, gn::AbstractFloat=inv(length(x)))
    h = Sequence(length(x))
    h[1] = step(params, x[1], h0, gn)
    for t = 2:length(x)
        h[t] = step(params, x[t], h[t-1], gn)
    end
    return h
end

@comp unfold(params::GatedRecurrentGradNorm, x::Vector, gn::AbstractFloat=inv(length(x))) = unfold(params, x, initial_state(params), gn)


"""
Convenience Constructor
"""
function GatedRecurrent(m::Int, n::Int; normed::Bool=false)
    wr, wz, wc = orthonormal(m, n), orthonormal(m, n), orthonormal(m, n)
    ur, uz, uc = orthonormal(m, m), orthonormal(m, m), orthonormal(m, m)
    br, bz, bc = ones(m, 1), zeros(m, 1), zeros(m, 1)
    h0 = zeros(m, 1)
    if normed
        return GatedRecurrentGradNorm(
            wr=wr, wz=wz, wc=wc, 
            ur=ur, uz=uz, uc=uc,
            br=br, bz=bz, bc=bc,
            h0=h0
        )
    else
        return GatedRecurrent(
            wr=wr, wz=wz, wc=wc, 
            ur=ur, uz=uz, uc=uc,
            br=br, bz=bz, bc=bc,
            h0=h0
        )
    end
end
