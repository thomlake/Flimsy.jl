
immutable GatedRecurrent <: RecurrentComponent1
    wr::GradVariable; wz::GradVariable; wc::GradVariable;
    ur::GradVariable; uz::GradVariable; uc::GradVariable;
    br::GradVariable; bz::GradVariable; bc::GradVariable;
    h_init::GradVariable;
    function GatedRecurrent(wr, wz, wc, ur, uz, uc, br, bz, bc, h_init)
        m, n = size(wr)
        size(wz) == (m, n) || error("Bad size(wz) == $(size(wz)) != ($m, $n)")
        size(wc) == (m, n) || error("Bad size(wc) == $(size(wc)) != ($m, $n)")
        size(ur) == (m, m) || error("Bad size(ur) == $(size(ur)) != ($m, $m)")
        size(uz) == (m, m) || error("Bad size(uz) == $(size(uz)) != ($m, $m)")
        size(uc) == (m, m) || error("Bad size(uc) == $(size(uc)) != ($m, $m)")
        size(br) == (m, 1) || error("Bad size(br) == $(size(br)) != ($m, 1)")
        size(bz) == (m, 1) || error("Bad size(bz) == $(size(bz)) != ($m, 1)")
        size(bc) == (m, 1) || error("Bad size(bc) == $(size(bc)) != ($m, 1)")
        size(h_init) == (m, 1) || error("Bad size(h_init) == $(size(h_init)) != ($m, 1)")
        return new(wr, wz, wc, ur, uz, uc, br, bz, bc, h_init)
    end
end

# GatedRecurrent{V<:Variable}(wr::V, wz::V, wc::V, ur::V, uz::V, uc::V, br::V, bz::V, bc::V, h0::V) = 
#     GatedRecurrent{V}(wr, wz, wc, ur, uz, uc, br, bz, bc, h0)

initial_state(scope::Scope, params::GatedRecurrent) = params.h_init

function Base.step(scope::Scope, params::GatedRecurrent, x, htm1)
    @with scope begin
        r = sigmoid(plus(linear(params.wr, x), linear(params.ur, htm1), params.br))
        z = sigmoid(plus(linear(params.wz, x), linear(params.uz, htm1), params.bz))
        c = tanh(plus(linear(params.wc, x), mult(r, linear(params.uc, htm1)), params.bc))
        return plus(mult(z, htm1), mult(minus(1.0, z), c))
    end
end

Base.step(scope::Scope, params::GatedRecurrent, x) = @with scope step(params, x, initial_state(params))

function unfold(scope::Scope, params::GatedRecurrent, x::Vector, h_init::Variable)
    @with scope begin
        h = Sequence(length(x))
        h[1] = step(params, x[1], h_init)
        for t = 2:length(x)
            h[t] = step(params, x[t], h[t-1])
        end
    end
    return h
end

unfold(scope::Scope, params::GatedRecurrent, x::Vector) = @with scope unfold(params, x, initial_state(params))

"""
GatedRecurrent Component with normalized hidden unit gradients.
By default gradients are normalized to 1/timesteps.
"""
immutable GatedRecurrentGradNorm <: RecurrentComponent1
    wr::GradVariable; wz::GradVariable; wc::GradVariable;
    ur::GradVariable; uz::GradVariable; uc::GradVariable;
    br::GradVariable; bz::GradVariable; bc::GradVariable;
    h_init::GradVariable;
    function GatedRecurrentGradNorm(wr, wz, wc, ur, uz, uc, br, bz, bc, h_init)
        m, n = size(wr)
        size(wz) == (m, n) || error("Bad size(wz) == $(size(wz)) != ($m, $n)")
        size(wc) == (m, n) || error("Bad size(wc) == $(size(wc)) != ($m, $n)")
        size(ur) == (m, m) || error("Bad size(ur) == $(size(ur)) != ($m, $m)")
        size(uz) == (m, m) || error("Bad size(uz) == $(size(uz)) != ($m, $m)")
        size(uc) == (m, m) || error("Bad size(uc) == $(size(uc)) != ($m, $m)")
        size(br) == (m, 1) || error("Bad size(br) == $(size(br)) != ($m, 1)")
        size(bz) == (m, 1) || error("Bad size(bz) == $(size(bz)) != ($m, 1)")
        size(bc) == (m, 1) || error("Bad size(bc) == $(size(bc)) != ($m, 1)")
        size(h_init) == (m, 1) || error("Bad size(h_init) == $(size(h_init)) != ($m, 1)")
        return new(wr, wz, wc, ur, uz, uc, br, bz, bc, h_init)
    end
end

# GatedRecurrentGradNorm{V<:Variable}(wr::V, wz::V, wc::V, ur::V, uz::V, uc::V, br::V, bz::V, bc::V, h0::V) = 
#     GatedRecurrentGradNorm{V}(wr, wz, wc, ur, uz, uc, br, bz, bc, h0)

initial_state(scope::Scope, params::GatedRecurrentGradNorm) = params.h_init

function Base.step(scope::Scope, params::GatedRecurrentGradNorm, x, htm1, gn::AbstractFloat=1.0)
    @with scope begin
        r = sigmoid(plus(linear(params.wr, x), linear(params.ur, htm1), params.br))
        z = sigmoid(plus(linear(params.wz, x), linear(params.uz, htm1), params.bz))
        c_pre = plus(linear(params.wc, x), mult(r, linear(params.uc, htm1)), params.bc)
        gradnorm(c_pre, gn)
        c = tanh(c_pre)
        return plus(mult(z, htm1), mult(minus(1.0, z), c))
    end
end

Base.step(scope::Scope, params::GatedRecurrentGradNorm, x, gn::AbstractFloat=1.0) = @with scope step(params, x, initial_state(params), gn)

function unfold(scope::Scope, params::GatedRecurrentGradNorm, x::Vector, h_init::Variable, gn::AbstractFloat=inv(length(x)))
    @with scope begin
        h = Sequence(length(x))
        h[1] = step(params, x[1], h_init, gn)
        for t = 2:length(x)
            h[t] = step(params, x[t], h[t-1], gn)
        end
        return h
    end
end

unfold(scope::Scope, params::GatedRecurrentGradNorm, x::Vector, gn::AbstractFloat=inv(length(x))) = @with scope unfold(params, x, initial_state(params), gn)


"""
Convenience Constructor
"""
function GatedRecurrent(m::Int, n::Int; normed::Bool=false)
    wr, wz, wc = orthonormal(m, n), orthonormal(m, n), orthonormal(m, n)
    ur, uz, uc = orthonormal(m, m), orthonormal(m, m), orthonormal(m, m)
    br, bz, bc = ones(m, 1), zeros(m, 1), zeros(m, 1)
    h_init = zeros(m, 1)
    if normed
        return GatedRecurrentGradNorm(
            wr=wr, wz=wz, wc=wc, 
            ur=ur, uz=uz, uc=uc,
            br=br, bz=bz, bc=bc,
            h_init=h_init
        )
    else
        return GatedRecurrent(
            wr=wr, wz=wz, wc=wc, 
            ur=ur, uz=uz, uc=uc,
            br=br, bz=bz, bc=bc,
            h_init=h_init
        )
    end
end
