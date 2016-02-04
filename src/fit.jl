
typealias CachePair{F} Tuple{Matrix{F},GradVariable{F}}

paramdict(theta::Component) = [name => zero(param.data) for (name, param) in getnamedparams(theta)]

function paired{F<:AbstractFloat}(theta::Component{GradVariable{F}})
    pairs = CachePair{F}[]
    for param in convert(Vector, theta)
        push!(pairs, (zero(param.data), param))
    end
    return pairs
end

function gradnorm(theta::Component)
    ss = 0.0
    for param in getparams(theta)
        for i in eachindex(param)
            ss += abs2(param.grad[i])
        end
    end
    return sqrt(ss)
end

function gradnorm{V<:GradVariable}(paramvec::Vector{V})
    ss = 0.0
    for param in paramvec
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
abstract GradientDescent <: Optimizer

type PlainGradientDescent{T<:AbstractFloat,V<:GradVariable} <: GradientDescent
    learning_rate::T
    paramvec::Vector{V}
end

function update!(opt::GradientDescent)
    lr = opt.learning_rate
    for param in opt.paramvec
        for i in eachindex(param)
            param.data[i] -= lr * param.grad[i]
        end
        fill!(param.grad, 0)
    end
end

type ScaledGradientDescent{T<:AbstractFloat,V<:GradVariable} <: GradientDescent
    learning_rate::T
    max_norm::T
    paramvec::Vector{V}
end

function update!(opt::ScaledGradientDescent)
    lr = opt.learning_rate
    gnorm = gradnorm(opt.paramvec)
    if gnorm > opt.max_norm
        # println("clip $gnorm => $(opt.max_norm)")
        lr *= opt.max_norm / gnorm
    end
    for param in opt.paramvec
        for i in eachindex(param)
            param.data[i] -= lr * param.grad[i]
        end
        fill!(param.grad, 0)
    end
end

type ClippedGradientDescent{T<:AbstractFloat,V<:GradVariable} <: GradientDescent
    learning_rate::T
    clip::T
    paramvec::Vector{V}
end

function update!(opt::ClippedGradientDescent, theta::Component)
    lr = opt.learning_rate
    ub = opt.clip
    lb = -opt.clip
    for param in opt.paramvec
        for i in eachindex(param)
            param.data[i] -= lr * max(min(param.grad[i], ub), lb)
        end
        fill!(param.grad, 0)
    end
end


####################
# --- Momentum --- #
####################
abstract Momentum <: Optimizer

type PlainMomentum{T<:AbstractFloat,F<:AbstractFloat} <: Momentum
    learning_rate::T
    momentum::T
    paired::Vector{CachePair{F}}
end

function update!(opt::Momentum, theta::Component)
    lr = opt.learning_rate
    mu = opt.momentum
    for (cache, param) in opt.paired
        @assert size(param) == size(cache)
        for i in eachindex(param)
            cache[i] = mu * cache[i] - lr * param.grad[i]
            param.data[i] += cache[i]
        end
        fill!(param.grad, 0)
    end
end

type ScaledMomentum{T<:AbstractFloat,V<:GradVariable} <: Momentum
    learning_rate::T
    momentum::T
    max_norm::T
    cache::Dict
end

function update!(opt::ScaledMomentum, theta::Component)
    lr = opt.learning_rate
    mu = opt.momentum
    gnorm = gradnorm(theta)
    if gnorm > opt.max_norm
        lr *= opt.max_norm / gnorm
    end
    for (name, param) in getnamedparams(theta)
        cache = opt.cache[name]
        @assert size(param) == size(cache)
        for i in eachindex(param)
            cache[i] = mu * cache[i] - lr * param.grad[i]
            param.data[i] += cache[i]
        end
        fill!(param.grad, 0)
    end
end

type ClippedMomentum{T<:AbstractFloat,V<:GradVariable} <: Momentum
    learning_rate::T
    momentum::T
    clip::T
    cache::Dict
end

function update!(opt::ClippedMomentum, theta::Component)
    lr = opt.learning_rate
    mu = opt.momentum
    ub = opt.clip
    lb = -opt.clip
    for (name, param) in getnamedparams(theta)
        cache = opt.cache[name]
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
abstract Nesterov <: Optimizer

type PlainNesterov{T<:AbstractFloat,V<:GradVariable} <: Nesterov
    learning_rate::T
    momentum::T
    cache::Dict
end

function update!(opt::Nesterov, theta::Component)
    lr = opt.learning_rate
    mu = opt.momentum
    upmu = 1 + mu
    for (name, param) in getnamedparams(theta)
        cache = opt.cache[name]
        @assert size(param) == size(cache)
        for i in eachindex(param)
            prev = cache[i]
            cache[i] = mu * cache[i] - lr * param.grad[i]
            param.data[i] += -mu * prev + upmu * cache[i]
        end
        fill!(param.grad, 0)
    end
end

type ScaledNesterov{T<:AbstractFloat,V<:GradVariable} <: Nesterov
    learning_rate::T
    momentum::T
    max_norm::T
    cache::Dict
end

function update!(opt::ScaledNesterov, theta::Component)
    lr = opt.learning_rate
    mu = opt.momentum
    upmu = 1 + mu
    gnorm = gradnorm(theta)
    if gnorm > opt.max_norm
        lr *= opt.max_norm / gnorm
    end
    for (name, param) in getnamedparams(theta)
        cache = opt.cache[name]
        @assert size(param) == size(cache)
        for i in eachindex(param)
            prev = cache[i]
            cache[i] = mu * cache[i] - lr * param.grad[i]
            param.data[i] += -mu * prev + upmu * cache[i]
        end
        fill!(param.grad, 0)
    end
end

type ClippedNesterov{T<:AbstractFloat,V<:GradVariable} <: Nesterov
    learning_rate::T
    momentum::T
    clip::T
    cache::Dict
end

function update!(opt::ClippedNesterov, theta::Component)
    lr = opt.learning_rate
    mu = opt.momentum
    upmu = 1 + mu
    ub = opt.clip
    lb = -opt.clip
    for (name, param) in getnamedparams(theta)
        cache = opt.cache[name]
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
abstract RMSProp <: Optimizer

type PlainRMSProp{T<:AbstractFloat,F<:AbstractFloat} <: RMSProp
    learning_rate::T
    decay::T
    pairs::Vector{CachePair{F}}
end

function update!(opt::RMSProp)
    lr = opt.learning_rate
    decay = opt.decay
    umdecay = 1 - decay
    for (cache, param) in opt.pairs
        @assert size(param) == size(cache)
        for i in eachindex(cache)
            cache[i] = decay * cache[i] + umdecay * param.grad[i] * param.grad[i]
            delta = param.grad[i] / sqrt(cache[i] + 1e-8)
            param.data[i] -= lr * delta
        end
        fill!(param.grad, 0)
    end
end

type ScaledRMSProp{T<:AbstractFloat,V<:GradVariable} <: RMSProp
    learning_rate::T
    decay::T
    max_norm::T
    cache::Dict
end

function update!(opt::ScaledRMSProp, theta::Component)
    lr = opt.learning_rate
    decay = opt.decay
    umdecay = 1 - decay
    gnorm = gradnorm(theta)
    scale = gnorm > opt.max_norm ? opt.max_norm / gnorm : 1.0
    for (name, param) in getnamedparams(theta)
        cache = opt.cache[name]
        @assert size(param) == size(cache)
        for i in eachindex(cache)
            grad = scale * param.grad[i]
            cache[i] = decay * cache[i] + umdecay * grad * grad
            delta = grad / sqrt(cache[i] + 1e-8)
            param.data[i] -= lr * delta
        end
        fill!(param.grad, 0)
    end
end

type ClippedRMSProp{T<:AbstractFloat,V<:GradVariable} <: RMSProp
    learning_rate::T
    decay::T
    clip::T
    cache::Dict
end

function update!(opt::ClippedRMSProp, theta::Component)
    lr = opt.learning_rate
    decay = opt.decay
    umdecay = 1 - decay
    ub = opt.clip
    lb = -opt.clip
    for (name, param) in getnamedparams(theta)
        cache = opt.cache[name]
        @assert size(param) == size(cache)
        for i in eachindex(cache)
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
abstract AdaDelta <: Optimizer

type PlainAdaDelta{T<:AbstractFloat,V<:GradVariable} <: AdaDelta
    decay::T
    grad_cache::Dict
    delta_cache::Dict
end

function update!(opt::AdaDelta, theta::Component)
    decay = opt.decay
    umdecay = 1 - decay
    for (name, param) in getnamedparams(theta)
        g2_tm1 = opt.grad_cache[name]
        d2_tm1 = opt.delta_cache[name]
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

type ScaledAdaDelta{T<:AbstractFloat,V<:GradVariable} <: AdaDelta
    decay::T
    max_norm::T
    grad_cache::Dict
    delta_cache::Dict
end

function update!(opt::ScaledAdaDelta, theta::Component)
    decay = opt.decay
    umdecay = 1 - decay
    gnorm = gradnorm(theta)
    scale = gnorm > opt.max_norm ? opt.max_norm / gnorm : 1.0
    for (name, param) in getnamedparams(theta)
        g2_tm1 = opt.grad_cache[name]
        d2_tm1 = opt.delta_cache[name]
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

type ClippedAdaDelta{T<:AbstractFloat,V<:GradVariable} <: AdaDelta
    decay::T
    clip::T
    grad_cache::Dict
    delta_cache::Dict
end

function update!(opt::ClippedAdaDelta, theta::Component)
    decay = opt.decay
    umdecay = 1 - decay
    ub = opt.clip
    lb = -opt.clip
    for (name, param) in getnamedparams(theta)
        g2_tm1 = opt.grad_cache[name]
        d2_tm1 = opt.delta_cache[name]
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

################
# --- Adam --- #
################
abstract Adam <: Optimizer

type PlainAdam{T<:AbstractFloat,V<:GradVariable} <: Adam
    learning_rate::T
    moment1_decay::T
    moment2_decay::T
    epsilon::T
    timestep::Int
    moment1_cache::Dict
    moment2_cache::Dict
end

function update!(opt::Adam, theta::Component)
    opt.timestep += 1
    lr = opt.learning_rate
    b1 = opt.moment1_decay
    b2 = opt.moment2_decay
    e = opt.epsilon
    t = opt.timestep
    umb1 = 1 - b1
    umb2 = 1 - b2
    umb1sqr = 1 - (b1 * b1)
    umb2sqr = 1 - (b2 * b2)

    for (name, param) in getnamedparams(theta)
        m1 = opt.moment1_cache[name]
        m2 = opt.moment2_cache[name]
        @assert size(param) == size(m1) == size(m2)
        for i in eachindex(param)
            g = param.grad[i]
            m1[i] = b1 * m1[i] + umb1 * g
            m2[i] = b2 * m2[i] + umb2 * g * g
            m1hat = m1[i] / umb1sqr
            m2hat = m2[i] / umb2sqr
            param.data[i] -= lr * m1hat / sqrt(m2hat + e)
        end
        fill!(param.grad, 0)
    end
end

type ScaledAdam{T<:AbstractFloat,V<:GradVariable} <: Adam
    learning_rate::T
    moment1_decay::T
    moment2_decay::T
    epsilon::T
    max_norm::T
    timestep::Int
    moment1_cache::Dict
    moment2_cache::Dict
end

function update!(opt::ScaledAdam, theta::Component)
    opt.timestep += 1
    lr = opt.learning_rate
    b1 = opt.moment1_decay
    b2 = opt.moment2_decay
    e = opt.epsilon
    t = opt.timestep
    umb1 = 1 - b1
    umb2 = 1 - b2
    umb1sqr = 1 - (b1 * b1)
    umb2sqr = 1 - (b2 * b2)
    gnorm = gradnorm(theta)
    scale = gnorm > opt.max_norm ? opt.max_norm / gnorm : 1.0
    for (name, param) in getnamedparams(theta)
        m1 = opt.moment1_cache[name]
        m2 = opt.moment2_cache[name]
        @assert size(param) == size(m1) == size(m2)
        for i in eachindex(param)
            g = scale * param.grad[i]
            m1[i] = b1 * m1[i] + umb1 * g
            m2[i] = b2 * m2[i] + umb2 * g * g
            m1hat = m1[i] / umb1sqr
            m2hat = m2[i] / umb2sqr
            param.data[i] -= lr * m1hat / sqrt(m2hat + e)
        end
        fill!(param.grad, 0)
    end
end

type ClippedAdam{T<:AbstractFloat,V<:GradVariable} <: Adam
    learning_rate::T
    moment1_decay::T
    moment2_decay::T
    epsilon::T
    clip::T
    timestep::Int
    moment1_cache::Dict
    moment2_cache::Dict
end

function update!(opt::ClippedAdaDelta, theta::Component)
    opt.timestep += 1
    lr = opt.learning_rate
    b1 = opt.moment1_decay
    b2 = opt.moment2_decay
    e = opt.epsilon
    t = opt.timestep
    umb1 = 1 - b1
    umb2 = 1 - b2
    umb1sqr = 1 - (b1 * b1)
    umb2sqr = 1 - (b2 * b2)
    ub = opt.clip
    lb = -opt.clip
    for (name, param) in getnamedparams(theta)
        m1 = opt.moment1_cache[name]
        m2 = opt.moment2_cache[name]
        @assert size(param) == size(m1) == size(m2)
        for i in eachindex(param)
            g = max(min(param.grad[i], ub), lb)
            m1[i] = b1 * m1[i] + umb1 * g
            m2[i] = b2 * m2[i] + umb2 * g * g
            m1hat = m1[i] / umb1sqr
            m2hat = m2[i] / umb2sqr
            param.data[i] -= lr * m1hat / sqrt(m2hat + e)
        end
        fill!(param.grad, 0)
    end
end

########################################################
# --- Convenience function for creating optimizers --- #
########################################################
function optimizer{O<:Optimizer}(::Type{O}, theta::Component;
    learning_rate::Real=0.1,
    clipping_type::Symbol=:none,
    clip::Real=5.0,
    decay::Real=0.9,
    moment1_decay::Real=0.9,
    moment2_decay::Real=0.999,
    epsilon::Real=1e-9,
    momentum::Real=0.87,
    )
    if O <: GradientDescent
        if clipping_type == :none
            return PlainGradientDescent(learning_rate, convert(Vector, theta))
        elseif clipping_type == :scale
            return ScaledGradientDescent(learning_rate, clip)
        elseif clipping_type == :clip
            return ClippedGradientDescent(learning_rate, clip)
        else
            error("unkown clipping_type: $clipping_type for optimizer: $O")
        end
    elseif O <: Momentum
        if clipping_type == :none
            return PlainMomentum(learning_rate, momentum, paramdict(theta))
        elseif clipping_type == :scale
            return ScaledMomentum(learning_rate, momentum, clip, paramdict(theta))
        elseif clipping_type == :clip
            return ClippedMomentum(learning_rate, momentum, clip, paramdict(theta))
        else
            error("unkown clipping_type: $clipping_type for optimizer: $O")
        end
    elseif O <: Nesterov
        if clipping_type == :none
            return PlainNesterov(learning_rate, momentum, paramdict(theta))
        elseif clipping_type == :scale
            return ScaledNesterov(learning_rate, momentum, clip, paramdict(theta))
        elseif clipping_type == :clip
            return ClippedNesterov(learning_rate, momentum, clip, paramdict(theta))
        else
            error("unkown clipping_type: $clipping_type for optimizer: $O")
        end
    elseif O <: RMSProp
        if clipping_type == :none
            return PlainRMSProp(learning_rate, decay, paired(theta))
        elseif clipping_type == :scale
            return ScaledRMSProp(learning_rate, decay, clip, paramdict(theta))
        elseif clipping_type == :clip
            return ClippedRMSProp(learning_rate, decay, clip, paramdict(theta))
        else
            error("unkown clipping_type: $clipping_type for optimizer: $O")
        end
    elseif O <: AdaDelta
        if clipping_type == :none
            return PlainAdaDelta(decay, paramdict(theta), paramdict(theta))
        elseif clipping_type == :scale
            return ScaledAdaDelta(decay, clip, paramdict(theta), paramdict(theta))
        elseif clipping_type == :clip
            return ClippedAdaDelta(decay, clip, paramdict(theta), paramdict(theta))
        else
            error("unkown clipping_type: $clipping_type for optimizer: $O")
        end
    elseif O <: Adam
        if clipping_type == :none
            return PlainAdam(learning_rate, moment1_decay, moment2_decay, epsilon, 0, paramdict(theta), paramdict(theta))
        elseif clipping_type == :scale
            return ScaledAdam(learning_rate, moment1_decay, moment2_decay, epsilon, clip, 0, paramdict(theta), paramdict(theta))
        elseif clipping_type == :clip
            return ClippedAdam(learning_rate, moment1_decay, moment2_decay, epsilon, clip, 0, paramdict(theta), paramdict(theta))
        else
            error("unkown clipping_type: $clipping_type for optimizer: $O")
        end
    else
        error("uknown optimizer: $O")
    end
end
