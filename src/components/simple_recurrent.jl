
"""
SimpleRecurrent Component.

Implements the hidden layer of a Simple Recurrent Neural Network (aka Elman Network).

### Update Equations
    h[1] = f(w * x[1] + u * h0 + b)
    h[t] = f(w * x[t] + u * h[t-1] + b)

### Fields
* `f::Function` Activation function.
* `w::Variable` Input to hidden weights.
* `u::Variable` Hidden to hidden weights.
* `b::Variable` Hidden bias.
* `h0::Variable` Initial state.
"""
immutable SimpleRecurrent{V<:Variable} <: RecurrentComponent1{V}
    f::Function
    w::V
    u::V
    b::V
    h0::V
    function SimpleRecurrent(f::Function, w::V, u::V, b::V, h0::V)
        m, n = size(w)
        size(u) == (m, m) || throw(DimensionMismatch("Bad size(u) == $(size(u)) != ($m, $m)"))
        size(b) == (m, 1) || throw(DimensionMismatch("Bad size(b) == $(size(b)) != ($m, 1)"))
        size(h0) == (m, 1) || throw(DimensionMismatch("Bad size(h0) == $(size(h0)) != ($m, 1)"))
        return new(f, w, u, b, h0)
    end
end
SimpleRecurrent{V<:Variable}(f::Function, w::V, u::V, b::V, h0::V) = SimpleRecurrent{V}(f, w, u, b, h0)

@component Base.step(p::SimpleRecurrent, x, htm1) = p.f(plus(linear(p.w, x), linear(p.u, htm1), p.b))

@component Base.step(p::SimpleRecurrent, x) = step(p, x, p.h0)

@component function unfold(p::SimpleRecurrent, x::Vector)
    h = Sequence(eltype(p), length(x))
    h[1] = step(p, x[1])
    for t = 2:length(x)
        h[t] = step(p, x[t], h[t-1])
    end 
    return h
end


"""
SimpleRecurrent component with normalized hidden unit gradients.
By default gradients are normalized to 1/timesteps.
"""
immutable SimpleRecurrentGradNorm{V<:Variable} <: RecurrentComponent1{V}
    f::Function
    w::V
    u::V
    b::V
    h0::V
    function SimpleRecurrentGradNorm(f::Function, w::V, u::V, b::V, h0::V)
        m, n = size(w)
        size(u) == (m, m) || throw(DimensionMismatch("Bad size(u) == $(size(u)) != ($m, $m)"))
        size(b) == (m, 1) || throw(DimensionMismatch("Bad size(b) == $(size(b)) != ($m, 1)"))
        size(h0) == (m, 1) || throw(DimensionMismatch("Bad size(h0) == $(size(h0)) != ($m, 1)"))
        return new(f, w, u, b, h0)
    end
end
SimpleRecurrentGradNorm{V<:Variable}(f::Function, w::V, u::V, b::V, h0::V) = SimpleRecurrentGradNorm{V}(f, w, u, b, h0)

@component function Base.step(params::SimpleRecurrentGradNorm, x, htm1, gn::AbstractFloat=1.0)
    h_pre = plus(linear(params.w, x), linear(params.u, htm1), params.b)
    gradnorm(h_pre, gn)
    return params.f(h_pre)
end

@component Base.step(params::SimpleRecurrentGradNorm, x, gn::AbstractFloat=1.0) = step(params, x, params.h0, gn)

@component function unfold(params::SimpleRecurrentGradNorm, x::Vector, gn::AbstractFloat=inv(length(x)))
    h = Sequence(eltype(params), length(x))
    h[1] = step(params, x[1], gn)
    for t = 2:length(x)
        h[t] = step(params, x[t], h[t-1], gn)
    end
    return h
end

"""
Convenience Constructor
"""
function SimpleRecurrent(m::Int, n::Int; f::Function=tanh, normed::Bool=false)
    w = orthonormal(f, m, n)
    u = orthonormal(f, m, m)
    b = zeros(m, 1)
    h0 = zeros(m, 1)

    if normed
        return SimpleRecurrentGradNorm(f=f, w=w, u=u, b=b, h0=h0)
    else
        return SimpleRecurrent(f=f, w=w, u=u, b=b, h0=h0)
    end
end

