paramdict(theta::Component) = [name => zero(param.data) for (name, param) in getnamedparams(theta)]

function gradnorm(theta::Component)
    ss = 0.0
    for param in getparams(theta)
        for i in eachindex(param)
            ss += abs2(param.grad[i])
        end
    end
    return sqrt(ss)
end

abstract Optimizer

#######################################
# --- Stochastic Gradient Descent --- #
#######################################
type SGD{T<:AbstractFloat} <: Optimizer
    learning_rate::T
end

function update!(sgd::SGD, theta::Component)
    lr = sgd.learning_rate
    for param in getparams(theta)
        for i in eachindex(param)
            param.data[i] -= lr * param.grad[i]
        end
        fill!(param.grad, 0)
    end
end

type ScaledSGD{T<:AbstractFloat} <: Optimizer
    learning_rate::T
    max_norm::T
end

function update!(sgd::ScaledSGD, theta::Component)
    lr = sgd.learning_rate
    gnorm = gradnorm(theta)
    if gnorm > sgd.max_norm
        lr *= sgd.max_norm / gnorm
    end
    for param in getparams(theta)
        for i in eachindex(param)
            param.data[i] -= lr * param.grad[i]
        end
        fill!(param.grad, 0)
    end
end

type ClippedSGD{T<:AbstractFloat} <: Optimizer
    learning_rate::T
    clip::T
end

function update!(sgd::ClippedSGD, theta::Component)
    lr = sgd.learning_rate
    ub = sgd.clip
    lb = -sgd.clip
    for param in getparams(theta)
        for i in eachindex(param)
            param.data[i] -= lr * max(min(param.grad[i], ub), lb)
        end
        fill!(param.grad, 0)
    end
end


####################
# --- Momentum --- #
####################
type Momentum{T<:AbstractFloat} <: Optimizer
    learning_rate::T
    momentum::T
    cache::Dict
end

function update!(m::Momentum, theta::Component)
    lr = m.learning_rate
    mu = m.momentum
    for (name, param) in getnamedparams(theta)
        cache = m.cache[name]
        @assert size(param) == size(cache)
        for i in eachindex(param)
            cache[i] = mu * cache[i] - lr * param.grad[i]
            param.data[i] += cache[i]
        end
        fill!(param.grad, 0)
    end
end

type ScaledMomentum{T<:AbstractFloat} <: Optimizer
    learning_rate::T
    momentum::T
    max_norm::T
    cache::Dict
end

function update!(m::ScaledMomentum, theta::Component)
    lr = m.learning_rate
    mu = m.momentum
    gnorm = gradnorm(theta)
    if gnorm > m.max_norm
        lr *= m.max_norm / gnorm
    end
    for (name, param) in getnamedparams(theta)
        cache = m.cache[name]
        @assert size(param) == size(cache)
        for i in eachindex(param)
            cache[i] = mu * cache[i] - lr * param.grad[i]
            param.data[i] += cache[i]
        end
        fill!(param.grad, 0)
    end
end

type ClippedMomentum{T<:AbstractFloat} <: Optimizer
    learning_rate::T
    momentum::T
    clip::T
    cache::Dict
end

function update!(m::ClippedMomentum, theta::Component)
    lr = m.learning_rate
    mu = m.momentum
    ub = m.clip
    lb = -m.clip
    for (name, param) in getnamedparams(theta)
        cache = m.cache[name]
        @assert size(param) == size(cache)
        for i in eachindex(param)
            cache[i] = mu * cache[i] - lr * max(min(param.grad[i], ub), lb)
            param.data[i] += cache[i]
        end
        fill!(param.grad, 0)
    end
end


#############################
# --- Nesterov Momentum --- #
#############################
type Nesterov{T<:AbstractFloat} <: Optimizer
    learning_rate::T
    momentum::T
    cache::Dict
end

function update!(m::Nesterov, theta::Component)
    lr = m.learning_rate
    mu = m.momentum
    upmu = 1 + mu
    for (name, param) in getnamedparams(theta)
        cache = m.cache[name]
        @assert size(param) == size(cache)
        for i in eachindex(param)
            prev = cache[i]
            cache[i] = mu * cache[i] - lr * param.grad[i]
            param.data[i] += -mu * prev + upmu * cache[i]
        end
        fill!(param.grad, 0)
    end
end

type ScaledNesterov{T<:AbstractFloat} <: Optimizer
    learning_rate::T
    momentum::T
    max_norm::T
    cache::Dict
end

function update!(m::ScaledNesterov, theta::Component)
    lr = m.learning_rate
    mu = m.momentum
    upmu = 1 + mu
    gnorm = gradnorm(theta)
    if gnorm > m.max_norm
        lr *= m.max_norm / gnorm
    end
    for (name, param) in getnamedparams(theta)
        cache = m.cache[name]
        @assert size(param) == size(cache)
        for i in eachindex(param)
            prev = cache[i]
            cache[i] = mu * cache[i] - lr * param.grad[i]
            param.data[i] += -mu * prev + upmu * cache[i]
        end
        fill!(param.grad, 0)
    end
end

type ClippedNesterov{T<:AbstractFloat} <: Optimizer
    learning_rate::T
    momentum::T
    clip::T
    cache::Dict
end

function update!(m::ClippedNesterov, theta::Component)
    lr = m.learning_rate
    mu = m.momentum
    upmu = 1 + mu
    ub = m.clip
    lb = -m.clip
    for (name, param) in getnamedparams(theta)
        cache = m.cache[name]
        @assert size(param) == size(cache)
        for i in eachindex(param)
            prev = cache[i]
            cache[i] = mu * cache[i] - lr * max(min(param.grad[i], ub), lb)
            param.data[i] += -mu * prev + upmu * cache[i]
        end
        fill!(param.grad, 0)
    end
end


####################
# --- RMS Prop --- #
####################
type RMSProp{T<:AbstractFloat} <: Optimizer
    learning_rate::T
    decay::T
    cache::Dict
end

function update!(r::RMSProp, theta::Component)
    lr = r.learning_rate
    decay = r.decay
    umdecay = 1 - decay
    for (name, param) in getnamedparams(theta)
        cache = r.cache[name]
        @assert size(param) == size(cache)
        for i in eachindex(cache)
            cache[i] = decay * cache[i] + umdecay * param.grad[i] * param.grad[i]
            delta = param.grad[i] / sqrt(cache[i] + 1e-8)
            param.data[i] -= lr * delta
        end
        fill!(param.grad, 0)
    end
end

type ScaledRMSProp{T<:AbstractFloat} <: Optimizer
    learning_rate::T
    decay::T
    max_norm::T
    cache::Dict
end

function update!(r::ScaledRMSProp, theta::Component)
    lr = r.learning_rate
    decay = r.decay
    umdecay = 1 - decay
    gnorm = gradnorm(theta)
    scale = gnorm > r.max_norm ? r.max_norm / gnorm : 1.0
    for (name, param) in getnamedparams(theta)
        cache = r.cache[name]
        @assert size(param) == size(cache)
        for i in eachindex(cache)
            # scale norm
            grad = scale * param.grad[i]
            cache[i] = decay * cache[i] + umdecay * grad * grad
            delta = grad / sqrt(cache[i] + 1e-8)
            param.data[i] -= lr * delta
        end
        fill!(param.grad, 0)
    end
end

type ClippedRMSProp{T<:AbstractFloat} <: Optimizer
    learning_rate::T
    decay::T
    clip::T
    cache::Dict
end

function update!(r::ClippedRMSProp, theta::Component)
    lr = r.learning_rate
    decay = r.decay
    umdecay = 1 - decay
    ub = r.clip
    lb = -r.clip
    for (name, param) in getnamedparams(theta)
        cache = r.cache[name]
        @assert size(param) == size(cache)
        for i in eachindex(cache)
            # clip grad
            grad = max(min(param.grad[i], ub), lb)
            cache[i] = decay * cache[i] + umdecay * grad * grad
            delta = grad / sqrt(cache[i] + 1e-8)
            param.data[i] -= lr * delta
        end
        fill!(param.grad, 0)
    end
end


####################
# --- AdaDelta --- #
####################
type AdaDelta{T<:AbstractFloat} <: Optimizer
    decay::T
    grad_cache::Dict
    delta_cache::Dict
end

function update!(a::AdaDelta, theta::Component)
    decay = a.decay
    umdecay = 1 - decay
    for (name, param) in getnamedparams(theta)
        g2_tm1 = a.grad_cache[name]
        d2_tm1 = a.delta_cache[name]
        @assert size(param) == size(g2_tm1) == size(d2_tm1)
        for i in eachindex(param)
            g = param.grad[i]
            g2_t = decay * g2_tm1[i] + umdecay * g * g
            delta = -sqrt((d2_tm1[i] + 1e-8) / (g2_t + 1e-8)) * g
            param.data[i] += delta
            d2_tm1[i] = decay * d2_tm1[i] + umdecay * delta * delta
            g2_tm1[i] = g2_t

        end
        fill!(param.grad, 0)
    end
end

type ScaledAdaDelta{T<:AbstractFloat} <: Optimizer
    decay::T
    max_norm::T
    grad_cache::Dict
    delta_cache::Dict
end

function update!(a::ScaledAdaDelta, theta::Component)
    decay = a.decay
    umdecay = 1 - decay
    gnorm = gradnorm(theta)
    scale = gnorm > r.max_norm ? r.max_norm / gnorm : 1.0
    for (name, param) in getnamedparams(theta)
        g2_tm1 = a.grad_cache[name]
        d2_tm1 = a.delta_cache[name]
        @assert size(param) == size(g2_tm1) == size(d2_tm1)
        for i in eachindex(param)
            g = scale * param.grad[i]
            g2_t = decay * g2_tm1[i] + umdecay * g * g
            delta = -sqrt((d2_tm1[i] + 1e-8) / (g2_t + 1e-8)) * g
            param.data[i] += delta
            d2_tm1[i] = decay * d2_tm1[i] + umdecay * delta * delta
            g2_tm1[i] = g2_t

        end
        fill!(param.grad, 0)
    end
end

type ClippedAdaDelta{T<:AbstractFloat} <: Optimizer
    decay::T
    clip::T
    grad_cache::Dict
    delta_cache::Dict
end

function update!(a::ClippedAdaDelta, theta::Component)
    decay = a.decay
    umdecay = 1 - decay
    ub = a.clip
    lb = -a.clip
    for (name, param) in getnamedparams(theta)
        g2_tm1 = a.grad_cache[name]
        d2_tm1 = a.delta_cache[name]
        @assert size(param) == size(g2_tm1) == size(d2_tm1)
        for i in eachindex(param)
            g = max(min(param.grad[i], ub), lb)
            g2_t = decay * g2_tm1[i] + umdecay * g * g
            delta = -sqrt((d2_tm1[i] + 1e-8) / (g2_t + 1e-8)) * g
            param.data[i] += delta
            d2_tm1[i] = decay * d2_tm1[i] + umdecay * delta * delta
            g2_tm1[i] = g2_t

        end
        fill!(param.grad, 0)
    end
end


# -- convenience function for creating optimizers -- #
function optimizer{O<:Optimizer}(::Type{O}, theta::Component;
    learning_rate::Real=0.1,
    clipping_type::Symbol=:none,
    clip::Real=5.0,
    decay::Real=0.9,
    momentum::Real=0.87,
    )
    if O <: SGD
        if clipping_type == :none
            return SGD(learning_rate)
        elseif clipping_type == :scale
            return ScaledSGD(learning_rate, clip)
        elseif clipping_type == :clip
            return ClippedSGD(learning_rate, clip)
        else
            error("unkown clipping_type: $clipping_type for optimizer: $O")
        end
    elseif O <: Momentum
        if clipping_type == :none
            return Momentum(learning_rate, momentum, paramdict(theta))
        elseif clipping_type == :scale
            return ScaledMomentum(learning_rate, momentum, clip, paramdict(theta))
        elseif clipping_type == :clip
            return ClippedMomentum(learning_rate, momentum, clip, paramdict(theta))
        else
            error("unkown clipping_type: $clipping_type for optimizer: $O")
        end
    elseif O <: Nesterov
        if clipping_type == :none
            return Nesterov(learning_rate, momentum, paramdict(theta))
        elseif clipping_type == :scale
            return ScaledNesterov(learning_rate, momentum, clip, paramdict(theta))
        elseif clipping_type == :clip
            return ClippedNesterov(learning_rate, momentum, clip, paramdict(theta))
        else
            error("unkown clipping_type: $clipping_type for optimizer: $O")
        end
    elseif O <: RMSProp
        if clipping_type == :none
            return RMSProp(learning_rate, decay, paramdict(theta))
        elseif clipping_type == :scale
            return ScaledRMSProp(learning_rate, decay, clip, paramdict(theta))
        elseif clipping_type == :clip
            return ClippedRMSProp(learning_rate, decay, clip, paramdict(theta))
        else
            error("unkown clipping_type: $clipping_type for optimizer: $O")
        end
    elseif O <: AdaDelta
        if clipping_type == :none
            return AdaDelta(decay, paramdict(theta), paramdict(theta))
        elseif clipping_type == :scale
            return ScaledAdaDelta(decay, clip, paramdict(theta), paramdict(theta))
        elseif clipping_type == :clip
            return ClippedAdaDelta(decay, clip, paramdict(theta), paramdict(theta))
        else
            error("unkown clipping_type: $clipping_type for optimizer: $O")
        end
    else
        error("uknown optimizer: $O")
    end
end
