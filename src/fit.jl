
# function gradnorm(theta::Component)
#     ss = 0.0
#     for param in getparams(theta)
#         for i in eachindex(param)
#             ss += abs2(param.grad[i])
#         end
#     end
#     return sqrt(ss)
# end

function gradnorm{V<:GradVariable}(paramvec::Vector{V})
    ss = 0.0
    for param in paramvec
        for i in eachindex(param)
            ss += abs2(param.grad[i])
        end
    end
    return sqrt(ss)
end

Cache{F<:AbstractFloat}(paramvec::Vector{GradVariable{F}}) = Matrix{F}[zero(p.data) for p in paramvec]

abstract Optimizer

#######################################
# --- Stochastic Gradient Descent --- #
#######################################
abstract GradientDescent <: Optimizer

type PlainGradientDescent{T<:AbstractFloat,F<:AbstractFloat} <: GradientDescent
    learning_rate::T
    paramvec::Vector{GradVariable{F}}
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

type ScaledGradientDescent{T<:AbstractFloat,F<:AbstractFloat} <: GradientDescent
    learning_rate::T
    max_norm::T
    paramvec::Vector{GradVariable{F}}
end

function update!(opt::ScaledGradientDescent)
    lr = opt.learning_rate
    gnorm = gradnorm(opt.paramvec)
    if gnorm > opt.max_norm
        lr *= opt.max_norm / gnorm
    end
    for param in opt.paramvec
        for i in eachindex(param)
            param.data[i] -= lr * param.grad[i]
        end
        fill!(param.grad, 0)
    end
end

type ClippedGradientDescent{T<:AbstractFloat,F<:AbstractFloat} <: GradientDescent
    learning_rate::T
    clip::T
    paramvec::Vector{GradVariable{F}}
end

function update!(opt::ClippedGradientDescent)
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
    paramvec::Vector{GradVariable{F}}
    cachevec::Vector{Matrix{F}}
end

function update!(opt::Momentum)
    lr = opt.learning_rate
    mu = opt.momentum
    for k = 1:length(opt.paramvec)
        param = opt.paramvec[k]
        cache = opt.cachevec[k]
        @assert size(param) == size(cache)
        for i in eachindex(param)
            cache[i] = mu * cache[i] - lr * param.grad[i]
            param.data[i] += cache[i]
        end
        fill!(param.grad, 0)
    end
end

type ScaledMomentum{T<:AbstractFloat,F<:AbstractFloat} <: Momentum
    learning_rate::T
    momentum::T
    max_norm::T
    paramvec::Vector{GradVariable{F}}
    cachevec::Vector{Matrix{F}}
end

function update!(opt::ScaledMomentum)
    lr = opt.learning_rate
    mu = opt.momentum
    gnorm = gradnorm(opt.paramvec)
    if gnorm > opt.max_norm
        lr *= opt.max_norm / gnorm
    end
    for k = 1:length(opt.paramvec)
        param = opt.paramvec[k]
        cache = opt.cachevec[k]
        @assert size(param) == size(cache)
        for i in eachindex(param)
            cache[i] = mu * cache[i] - lr * param.grad[i]
            param.data[i] += cache[i]
        end
        fill!(param.grad, 0)
    end
end

type ClippedMomentum{T<:AbstractFloat,F<:AbstractFloat} <: Momentum
    learning_rate::T
    momentum::T
    clip::T
    paramvec::Vector{GradVariable{F}}
    cachevec::Vector{Matrix{F}}
end

function update!(opt::ClippedMomentum)
    lr = opt.learning_rate
    mu = opt.momentum
    ub = opt.clip
    lb = -opt.clip
    for k = 1:length(opt.paramvec)
        param = opt.paramvec[k]
        cache = opt.cachevec[k]
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

type PlainNesterov{T<:AbstractFloat,F<:AbstractFloat} <: Nesterov
    learning_rate::T
    momentum::T
    paramvec::Vector{GradVariable{F}}
    cachevec::Vector{Matrix{F}}
end

function update!(opt::Nesterov)
    lr = opt.learning_rate
    mu = opt.momentum
    upmu = 1 + mu
    for k = 1:length(opt.paramvec)
        param = opt.paramvec[k]
        cache = opt.cachevec[k]
        @assert size(param) == size(cache)
        for i in eachindex(param)
            prev = cache[i]
            cache[i] = mu * cache[i] - lr * param.grad[i]
            param.data[i] += -mu * prev + upmu * cache[i]
        end
        fill!(param.grad, 0)
    end
end

type ScaledNesterov{T<:AbstractFloat,F<:AbstractFloat} <: Nesterov
    learning_rate::T
    momentum::T
    max_norm::T
    paramvec::Vector{GradVariable{F}}
    cachevec::Vector{Matrix{F}}
end

function update!(opt::ScaledNesterov)
    lr = opt.learning_rate
    mu = opt.momentum
    upmu = 1 + mu
    gnorm = gradnorm(opt.paramvec)
    if gnorm > opt.max_norm
        lr *= opt.max_norm / gnorm
    end
    for k = 1:length(opt.paramvec)
        param = opt.paramvec[k]
        cache = opt.cachevec[k]
        @assert size(param) == size(cache)
        for i in eachindex(param)
            prev = cache[i]
            cache[i] = mu * cache[i] - lr * param.grad[i]
            param.data[i] += -mu * prev + upmu * cache[i]
        end
        fill!(param.grad, 0)
    end
end

type ClippedNesterov{T<:AbstractFloat,F<:AbstractFloat} <: Nesterov
    learning_rate::T
    momentum::T
    clip::T
    paramvec::Vector{GradVariable{F}}
    cachevec::Vector{Matrix{F}}
end

function update!(opt::ClippedNesterov)
    lr = opt.learning_rate
    mu = opt.momentum
    upmu = 1 + mu
    ub = opt.clip
    lb = -opt.clip
    for k = 1:length(opt.paramvec)
        param = opt.paramvec[k]
        cache = opt.cachevec[k]
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
    paramvec::Vector{GradVariable{F}}
    cachevec::Vector{Matrix{F}}
end

function update!(opt::RMSProp)
    lr = opt.learning_rate
    decay = opt.decay
    umdecay = 1 - decay
    for k = 1:length(opt.paramvec)
        param = opt.paramvec[k]
        cache = opt.cachevec[k]
        @assert size(param) == size(cache)
        for i in eachindex(cache)
            cache[i] = decay * cache[i] + umdecay * param.grad[i] * param.grad[i]
            delta = param.grad[i] / sqrt(cache[i] + 1e-8)
            param.data[i] -= lr * delta
        end
        fill!(param.grad, 0)
    end
end

type ScaledRMSProp{T<:AbstractFloat,F<:AbstractFloat} <: RMSProp
    learning_rate::T
    decay::T
    max_norm::T
    paramvec::Vector{GradVariable{F}}
    cachevec::Vector{Matrix{F}}
end

function update!(opt::ScaledRMSProp)
    lr = opt.learning_rate
    decay = opt.decay
    umdecay = 1 - decay
    gnorm = gradnorm(opt.paramvec)
    scale = gnorm > opt.max_norm ? opt.max_norm / gnorm : 1.0
    
    for k = 1:length(opt.paramvec)
        param = opt.paramvec[k]
        cache = opt.cachevec[k]
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

type ClippedRMSProp{T<:AbstractFloat,F<:AbstractFloat} <: RMSProp
    learning_rate::T
    decay::T
    clip::T
    paramvec::Vector{GradVariable{F}}
    cachevec::Vector{Matrix{F}}
end

function update!(opt::ClippedRMSProp)
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

type PlainAdaDelta{T<:AbstractFloat,F<:AbstractFloat} <: AdaDelta
    decay::T
    paramvec::Vector{GradVariable{F}}
    cachevec::Vector{Matrix{F}}
    deltavec::Vector{Matrix{F}}
end

function update!(opt::AdaDelta)
    decay = opt.decay
    umdecay = 1 - decay
    for i in eachindex(opt.paramvec)
        param = opt.paramvec[i]
        g2_tm1 = opt.cachevec[i]
        d2_tm1 = opt.deltavec[i]
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

type ScaledAdaDelta{T<:AbstractFloat,F<:AbstractFloat} <: AdaDelta
    decay::T
    max_norm::T
    paramvec::Vector{GradVariable{F}}
    cachevec::Vector{Matrix{F}}
    deltavec::Vector{Matrix{F}}
end

function update!(opt::ScaledAdaDelta)
    decay = opt.decay
    umdecay = 1 - decay
    gnorm = gradnorm(opt.paramvec)
    scale = gnorm > opt.max_norm ? opt.max_norm / gnorm : 1.0

    for i in eachindex(opt.paramvec)
        param = opt.paramvec[i]
        g2_tm1 = opt.cachevec[i]
        d2_tm1 = opt.deltavec[i]
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

type ClippedAdaDelta{T<:AbstractFloat,F<:AbstractFloat} <: AdaDelta
    decay::T
    clip::T
    paramvec::Vector{GradVariable{F}}
    cachevec::Vector{Matrix{F}}
    deltavec::Vector{Matrix{F}}
end

function update!(opt::ClippedAdaDelta)
    decay = opt.decay
    umdecay = 1 - decay
    ub = opt.clip
    lb = -opt.clip

    for i in eachindex(opt.paramvec)
        param = opt.paramvec[i]
        g2_tm1 = opt.cachevec[i]
        d2_tm1 = opt.deltavec[i]
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

type PlainAdam{T<:AbstractFloat,F<:AbstractFloat} <: Adam
    learning_rate::T
    moment1_decay::T
    moment2_decay::T
    epsilon::T
    timestep::Int
    paramvec::Vector{GradVariable{F}}
    moment1_vec::Vector{Matrix{F}}
    moment2_vec::Vector{Matrix{F}}
end

function update!(opt::Adam)
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

    for i in eachindex(opt.paramvec)
        param = opt.paramvec[i]
        m1 = opt.moment1_vec[i]
        m2 = opt.moment2_vec[i]
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

type ScaledAdam{T<:AbstractFloat,F<:AbstractFloat} <: Adam
    learning_rate::T
    moment1_decay::T
    moment2_decay::T
    epsilon::T
    max_norm::T
    timestep::Int
    paramvec::Vector{GradVariable{F}}
    moment1_vec::Vector{Matrix{F}}
    moment2_vec::Vector{Matrix{F}}
end

function update!(opt::ScaledAdam)
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
    gnorm = gradnorm(opt.paramvec)
    scale = gnorm > opt.max_norm ? opt.max_norm / gnorm : 1.0

    for i in eachindex(opt.paramvec)
        param = opt.paramvec[i]
        m1 = opt.moment1_vec[i]
        m2 = opt.moment2_vec[i]
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

type ClippedAdam{T<:AbstractFloat,F<:AbstractFloat} <: Adam
    learning_rate::T
    moment1_decay::T
    moment2_decay::T
    epsilon::T
    clip::T
    timestep::Int
    paramvec::Vector{GradVariable{F}}
    moment1_vec::Vector{Matrix{F}}
    moment2_vec::Vector{Matrix{F}}
end

function update!(opt::ClippedAdaDelta)
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

    for i in eachindex(opt.paramvec)
        param = opt.paramvec[i]
        m1 = opt.moment1_vec[i]
        m2 = opt.moment2_vec[i]
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
        paramvec = convert(Vector, theta)
        if clipping_type == :none
            return PlainGradientDescent(learning_rate, paramvec)
        elseif clipping_type == :scale
            return ScaledGradientDescent(learning_rate, clip, paramvec)
        elseif clipping_type == :clip
            return ClippedGradientDescent(learning_rate, clip, paramvec)
        else
            error("unkown clipping_type: $clipping_type for optimizer: $O")
        end
    elseif O <: Momentum
        paramvec = convert(Vector, theta)
        cachevec = Cache(paramvec)
        if clipping_type == :none
            return PlainMomentum(learning_rate, momentum, paramvec, cachevec)
        elseif clipping_type == :scale
            return ScaledMomentum(learning_rate, momentum, clip, paramvec, cachevec)
        elseif clipping_type == :clip
            return ClippedMomentum(learning_rate, momentum, clip, paramvec, cachevec)
        else
            error("unkown clipping_type: $clipping_type for optimizer: $O")
        end
    elseif O <: Nesterov
        paramvec = convert(Vector, theta)
        cachevec = Cache(paramvec)
        if clipping_type == :none
            return PlainNesterov(learning_rate, momentum, paramvec, cachevec)
        elseif clipping_type == :scale
            return ScaledNesterov(learning_rate, momentum, clip, paramvec, cachevec)
        elseif clipping_type == :clip
            return ClippedNesterov(learning_rate, momentum, clip, paramvec, cachevec)
        else
            error("unkown clipping_type: $clipping_type for optimizer: $O")
        end
    elseif O <: RMSProp
        paramvec = convert(Vector, theta)
        cachevec = Cache(paramvec)
        if clipping_type == :none
            return PlainRMSProp(learning_rate, decay, paramvec, cachevec)
        elseif clipping_type == :scale
            return ScaledRMSProp(learning_rate, decay, clip, paramvec, cachevec)
        elseif clipping_type == :clip
            return ClippedRMSProp(learning_rate, decay, clip, paramvec, cachevec)
        else
            error("unkown clipping_type: $clipping_type for optimizer: $O")
        end
    elseif O <: AdaDelta
        paramvec = convert(Vector, theta)
        cachevec = Cache(paramvec)
        deltavec = Cache(paramvec)
        if clipping_type == :none
            return PlainAdaDelta(decay, paramvec, cachevec, deltavec)
        elseif clipping_type == :scale
            return ScaledAdaDelta(decay, clip, paramvec, cachevec, deltavec)
        elseif clipping_type == :clip
            return ClippedAdaDelta(decay, clip, paramvec, cachevec, deltavec)
        else
            error("unkown clipping_type: $clipping_type for optimizer: $O")
        end
    elseif O <: Adam
        paramvec = convert(Vector, theta)
        moment1_vec = Cache(paramvec)
        moment2_vec = Cache(paramvec)
        if clipping_type == :none
            return PlainAdam(learning_rate, moment1_decay, moment2_decay, epsilon, 0, paramvec, moment1_vec, moment2_vec)
        elseif clipping_type == :scale
            return ScaledAdam(learning_rate, moment1_decay, moment2_decay, epsilon, clip, 0, paramvec, moment1_vec, moment2_vec)
        elseif clipping_type == :clip
            return ClippedAdam(learning_rate, moment1_decay, moment2_decay, epsilon, clip, 0, paramvec, moment1_vec, moment2_vec)
        else
            error("unkown clipping_type: $clipping_type for optimizer: $O")
        end
    else
        error("uknown optimizer: $O")
    end
end
