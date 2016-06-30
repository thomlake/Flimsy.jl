
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
* `h::Variable` Initial state.
"""
immutable SimpleRecurrent{F<:Activation} <: RecurrentComponent1
    f::F
    w::GradVariable
    u::GradVariable
    b::GradVariable
    h_init::GradVariable
    function SimpleRecurrent(f::F, w::GradVariable, u::GradVariable, b::GradVariable, h_init::GradVariable)
        m, n = size(w)
        size(u) == (m, m) || throw(DimensionMismatch("Bad size(u) == $(size(u)) != ($m, $m)"))
        size(b) == (m, 1) || throw(DimensionMismatch("Bad size(b) == $(size(b)) != ($m, 1)"))
        size(h_init) == (m, 1) || throw(DimensionMismatch("Bad size(h_init) == $(size(h_init)) != ($m, 1)"))
        return new(f, w, u, b, h_init)
    end
end

SimpleRecurrent{F<:Activation}(f::F, w, u, b, h) = SimpleRecurrent{F}(f, w, u, b, h)

initial_state(scope::Scope, params::SimpleRecurrent) = params.h_init

Base.step(scope::Scope, p::SimpleRecurrent, x, h) = @with scope begin
    activate(p.f, plus(linear(p.w, x), linear(p.u, h), p.b))
end

Base.step(scope::Scope, p::SimpleRecurrent, x) = @with scope step(p, x, initial_state(p))

function unfold(scope::Scope, p::SimpleRecurrent, x::Vector, h_init::Variable) 
    @with scope begin
        h = Sequence(length(x))
        h[1] = step(p, x[1], h_init)
        for t = 2:length(x)
            h[t] = step(p, x[t], h[t-1])
        end 
        return h
    end
end

unfold(scope::Scope, p::SimpleRecurrent, x::Vector) = @with scope unfold(p, x, initial_state(p))


"""
SimpleRecurrent component with normalized hidden unit gradients.
By default gradients are normalized to 1/timesteps.
"""
immutable SimpleRecurrentGradNorm{F<:Activation} <: RecurrentComponent1
    f::F
    w::GradVariable
    u::GradVariable
    b::GradVariable
    h_init::GradVariable
    function SimpleRecurrentGradNorm(f::F, w::GradVariable, u::GradVariable, b::GradVariable, h_init::GradVariable)
        m, n = size(w)
        size(u) == (m, m) || throw(DimensionMismatch("Bad size(u) == $(size(u)) != ($m, $m)"))
        size(b) == (m, 1) || throw(DimensionMismatch("Bad size(b) == $(size(b)) != ($m, 1)"))
        size(h_init) == (m, 1) || throw(DimensionMismatch("Bad size(h_init) == $(size(h_init)) != ($m, 1)"))
        return new(f, w, u, b, h_init)
    end
end

SimpleRecurrentGradNorm{F<:Activation}(f::F, w::GradVariable, u::GradVariable, b::GradVariable, h_init::GradVariable) = SimpleRecurrentGradNorm{F}(f, w, u, b, h_init)

initial_state(scope::Scope, p::SimpleRecurrentGradNorm) = p.h_init

function Base.step(scope, p::SimpleRecurrentGradNorm, x, h, gn::AbstractFloat=1.0)
    @with scope begin
        h = activate(p.f, plus(linear(p.w, x), linear(p.u, h), p.b))
        gradnorm(h, gn)
        return h
    end
end

Base.step(scope::Scope, p::SimpleRecurrentGradNorm, x, gn::AbstractFloat=1.0) = @with scope step(p, x, initial_state(p), gn)

function unfold(scope::Scope, p::SimpleRecurrentGradNorm, x::Vector, h_init::Variable, gn::AbstractFloat=inv(length(x)))
    @with scope begin
        h = Sequence(length(x))
        h[1] = step(p, x[1], h_init, gn)
        for t = 2:length(x)
            h[t] = step(p, x[t], h[t-1], gn)
        end
        return h
    end
end

unfold(scope::Scope, p::SimpleRecurrentGradNorm, x::Vector, gn::AbstractFloat=inv(length(x))) = @with scope unfold(p, x, initial_state(p), gn)


"""
Convenience Constructor
"""
function SimpleRecurrent(m::Int, n::Int; normed::Bool=false, f::Activation=Tanh())
    w = orthonormal(tanh, m, n)
    u = orthonormal(tanh, m, m)
    b = zeros(m, 1)
    h_init = zeros(m, 1)

    if normed
        return SimpleRecurrentGradNorm(f=f, w=w, u=u, b=b, h_init=h_init)
    else
        return SimpleRecurrent(f=f, w=w, u=u, b=b, h_init=h_init)
    end
end
