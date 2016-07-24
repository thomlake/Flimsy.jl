
function norm_param_grad(paramvec::Vector{Variable})
    ss = 0.0
    for param in paramvec
        for i in eachindex(param)
            ss += abs2(param.grad[i])
        end
    end
    return sqrt(ss)
end

Cache(paramvec::Vector{Variable}) = Matrix{FloatX}[zero(p.data) for p in paramvec]

abstract Optimizer

#######################################
# --- Stochastic Gradient Descent --- #
#######################################
abstract GradientDescent <: Optimizer

# Plain
type PlainGradientDescent{T<:AbstractFloat} <: GradientDescent
    learning_rate::T
    paramvec::Vector{Variable}
end
function Base.show(io::IO, opt::PlainGradientDescent)
    p = [
        string("learning_rate=", opt.learning_rate),
    ]
    print(io, "PlainGradientDescent(", join(p, ", "), ")")
end

function update!(opt::GradientDescent)
    lr = opt.learning_rate
    for param in opt.paramvec
        @flimsy_inbounds for i in eachindex(param)
            param.data[i] -= lr * param.grad[i]
        end
        fill!(param.grad, 0)
    end
end

# Scale
type ScaledGradientDescent{T<:AbstractFloat} <: GradientDescent
    learning_rate::T
    max_norm::T
    paramvec::Vector{Variable}
end

function Base.show(io::IO, opt::ScaledGradientDescent)
    p = [
        string("learning_rate=", opt.learning_rate),
        string("max_norm=", opt.max_norm),
    ]
    print(io, "ScaledGradientDescent(", join(p, ", "), ")")
end

function update!(opt::ScaledGradientDescent)
    lr = opt.learning_rate
    gnorm = norm_param_grad(opt.paramvec)
    if gnorm > opt.max_norm
        lr *= opt.max_norm / gnorm
    end
    for param in opt.paramvec
        @flimsy_inbounds for i in eachindex(param)
            param.data[i] -= lr * param.grad[i]
        end
        fill!(param.grad, 0)
    end
end

# Clip
type ClippedGradientDescent{T<:AbstractFloat} <: GradientDescent
    learning_rate::T
    clip::T
    paramvec::Vector{Variable}
end

function Base.show(io::IO, opt::ClippedGradientDescent)
    p = [
        string("learning_rate=", opt.learning_rate),
        string("clip=", opt.clip),
    ]
    print(io, "ClippedGradientDescent(", join(p, ", "), ")")
end

function update!(opt::ClippedGradientDescent)
    lr = opt.learning_rate
    ub = opt.clip
    lb = -opt.clip
    for param in opt.paramvec
        @flimsy_inbounds for i in eachindex(param)
            param.data[i] -= lr * max(min(param.grad[i], ub), lb)
        end
        fill!(param.grad, 0)
    end
end


####################
# --- Momentum --- #
####################
abstract Momentum <: Optimizer

# Plain
type PlainMomentum{T<:AbstractFloat} <: Momentum
    learning_rate::T
    momentum::T
    paramvec::Vector{Variable}
    cachevec::Vector{Matrix{FloatX}}
end

function Base.show(io::IO, opt::PlainMomentum)
    p = [
        string("learning_rate=", opt.learning_rate),
        string("momentum=", opt.momentum),
    ]
    print(io, "PlainMomentum(", join(p, ", "), ")")
end

function update!(opt::Momentum)
    lr = opt.learning_rate
    mu = opt.momentum
    for k = 1:length(opt.paramvec)
        param = opt.paramvec[k]
        cache = opt.cachevec[k]
        @assert size(param) == size(cache)
        @flimsy_inbounds for i in eachindex(param)
            cache[i] = mu * cache[i] - lr * param.grad[i]
            param.data[i] += cache[i]
        end
        fill!(param.grad, 0)
    end
end

# Scale
type ScaledMomentum{T<:AbstractFloat} <: Momentum
    learning_rate::T
    momentum::T
    max_norm::T
    paramvec::Vector{Variable}
    cachevec::Vector{Matrix{FloatX}}
end

function Base.show(io::IO, opt::ScaledMomentum)
    p = [
        string("learning_rate=", opt.learning_rate),
        string("momentum=", opt.momentum),
        string("max_norm=", opt.max_norm),
    ]
    print(io, "ScaledMomentum(", join(p, ", "), ")")
end

function update!(opt::ScaledMomentum)
    lr = opt.learning_rate
    mu = opt.momentum
    gnorm = norm_param_grad(opt.paramvec)
    if gnorm > opt.max_norm
        lr *= opt.max_norm / gnorm
    end
    for k = 1:length(opt.paramvec)
        param = opt.paramvec[k]
        cache = opt.cachevec[k]
        @assert size(param) == size(cache)
        @flimsy_inbounds for i in eachindex(param)
            cache[i] = mu * cache[i] - lr * param.grad[i]
            param.data[i] += cache[i]
        end
        fill!(param.grad, 0)
    end
end

# Clip
type ClippedMomentum{T<:AbstractFloat} <: Momentum
    learning_rate::T
    momentum::T
    clip::T
    paramvec::Vector{Variable}
    cachevec::Vector{Matrix{FloatX}}
end

function Base.show(io::IO, opt::ClippedMomentum)
    p = [
        string("learning_rate=", opt.learning_rate),
        string("momentum=", opt.momentum),
        string("clip=", opt.clip),
    ]
    print(io, "ClippedMomentum(", join(p, ", "), ")")
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
        @flimsy_inbounds for i in eachindex(param)
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

# Plain
type PlainNesterov{T<:AbstractFloat} <: Nesterov
    learning_rate::T
    momentum::T
    paramvec::Vector{Variable}
    cachevec::Vector{Matrix{FloatX}}
end

function Base.show(io::IO, opt::PlainNesterov)
    p = [
        string("learning_rate=", opt.learning_rate),
        string("momentum=", opt.momentum),
    ]
    print(io, "PlainNesterov(", join(p, ", "), ")")
end

function update!(opt::Nesterov)
    lr = opt.learning_rate
    mu = opt.momentum
    upmu = 1 + mu
    for k = 1:length(opt.paramvec)
        param = opt.paramvec[k]
        cache = opt.cachevec[k]
        @assert size(param) == size(cache)
        @flimsy_inbounds for i in eachindex(param)
            prev = cache[i]
            cache[i] = mu * cache[i] - lr * param.grad[i]
            param.data[i] += -mu * prev + upmu * cache[i]
        end
        fill!(param.grad, 0)
    end
end

# Scale
type ScaledNesterov{T<:AbstractFloat} <: Nesterov
    learning_rate::T
    momentum::T
    max_norm::T
    paramvec::Vector{Variable}
    cachevec::Vector{Matrix{FloatX}}
end

function Base.show(io::IO, opt::ScaledNesterov)
    p = [
        string("learning_rate=", opt.learning_rate),
        string("momentum=", opt.momentum),
        string("max_norm=", opt.max_norm),
    ]
    print(io, "ScaledNesterov(", join(p, ", "), ")")
end

function update!(opt::ScaledNesterov)
    lr = opt.learning_rate
    mu = opt.momentum
    upmu = 1 + mu
    gnorm = norm_param_grad(opt.paramvec)
    if gnorm > opt.max_norm
        lr *= opt.max_norm / gnorm
    end
    for k = 1:length(opt.paramvec)
        param = opt.paramvec[k]
        cache = opt.cachevec[k]
        @assert size(param) == size(cache)
        @flimsy_inbounds for i in eachindex(param)
            prev = cache[i]
            cache[i] = mu * cache[i] - lr * param.grad[i]
            param.data[i] += -mu * prev + upmu * cache[i]
        end
        fill!(param.grad, 0)
    end
end

# Clip
type ClippedNesterov{T<:AbstractFloat} <: Nesterov
    learning_rate::T
    momentum::T
    clip::T
    paramvec::Vector{Variable}
    cachevec::Vector{Matrix{FloatX}}
end

function Base.show(io::IO, opt::ClippedNesterov)
    p = [
        string("learning_rate=", opt.learning_rate),
        string("momentum=", opt.momentum),
        string("clip=", opt.clip),
    ]
    print(io, "ClippedNesterov(", join(p, ", "), ")")
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
        @flimsy_inbounds for i in eachindex(param)
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
abstract RmsProp <: Optimizer

# Plain
type PlainRmsProp{T<:AbstractFloat} <: RmsProp
    learning_rate::T
    decay::T
    paramvec::Vector{Variable}
    cachevec::Vector{Matrix{FloatX}}
end

function Base.show(io::IO, opt::PlainRmsProp)
    p = [
        string("learning_rate=", opt.learning_rate),
        string("decay=", opt.decay),
    ]
    print(io, "PlainRmsProp(", join(p, ", "), ")")
end

function update!(opt::RmsProp)
    lr = opt.learning_rate
    decay = opt.decay
    umdecay = 1 - decay
    for k = 1:length(opt.paramvec)
        param = opt.paramvec[k]
        cache = opt.cachevec[k]
        @assert size(param) == size(cache)
        @flimsy_inbounds for i in eachindex(cache)
            cache[i] = decay * cache[i] + umdecay * param.grad[i] * param.grad[i]
            delta = param.grad[i] / sqrt(cache[i] + 1e-8)
            param.data[i] -= lr * delta
        end
        fill!(param.grad, 0)
    end
end

# Scale
type ScaledRmsProp{T<:AbstractFloat} <: RmsProp
    learning_rate::T
    decay::T
    max_norm::T
    paramvec::Vector{Variable}
    cachevec::Vector{Matrix{FloatX}}
end

function Base.show(io::IO, opt::ScaledRmsProp)
    p = [
        string("learning_rate=", opt.learning_rate),
        string("decay=", opt.decay),
        string("max_norm=", opt.max_norm),

    ]
    print(io, "ScaledRmsProp(", join(p, ", "), ")")
end

function update!(opt::ScaledRmsProp)
    lr = opt.learning_rate
    decay = opt.decay
    umdecay = 1 - decay
    gnorm = norm_param_grad(opt.paramvec)
    scale = gnorm > opt.max_norm ? opt.max_norm / gnorm : 1.0
    
    for k = 1:length(opt.paramvec)
        param = opt.paramvec[k]
        cache = opt.cachevec[k]
        @assert size(param) == size(cache)
        @flimsy_inbounds for i in eachindex(cache)
            grad = scale * param.grad[i]
            cache[i] = decay * cache[i] + umdecay * grad * grad
            delta = grad / sqrt(cache[i] + 1e-8)
            param.data[i] -= lr * delta
        end
        fill!(param.grad, 0)
    end
end

# Clip
type ClippedRmsProp{T<:AbstractFloat} <: RmsProp
    learning_rate::T
    decay::T
    clip::T
    paramvec::Vector{Variable}
    cachevec::Vector{Matrix{FloatX}}
end

function Base.show(io::IO, opt::ClippedRmsProp)
    p = [
        string("learning_rate=", opt.learning_rate),
        string("decay=", opt.decay),
        string("clip=", opt.clip),

    ]
    print(io, "ClippedRmsProp(", join(p, ", "), ")")
end

function update!(opt::ClippedRmsProp)
    lr = opt.learning_rate
    decay = opt.decay
    umdecay = 1 - decay
    ub = opt.clip
    lb = -opt.clip
    for k = 1:length(opt.paramvec)
        param = opt.paramvec[k]
        cache = opt.cachevec[k]
        @assert size(param) == size(cache)
        @flimsy_inbounds for i in eachindex(cache)
            grad = max(min(param.grad[i], ub), lb)
            cache[i] = decay * cache[i] + umdecay * grad * grad
            delta = grad / sqrt(cache[i] + 1e-8)
            param.data[i] -= lr * delta
        end
        fill!(param.grad, 0)
    end
end

######################
# --- Graves --- #
######################
"""
Generating Sequences With Recurrent Neural Networks
Alex Graves, http://arxiv.org/abs/1308.0850
"""
abstract Graves <: Optimizer

type PlainGraves{T<:AbstractFloat} <: Graves
    learning_rate::T
    momentum::T
    decay::T
    epsilon::T
    paramvec::Vector{Variable}
    cachevec::Vector{Matrix{FloatX}}
    m1vec::Vector{Matrix{FloatX}}
    m2vec::Vector{Matrix{FloatX}}
end

function Base.show(io::IO, opt::PlainGraves)
    p = [
        string("learning_rate=", opt.learning_rate),
        string("momentum=", opt.momentum),
        string("decay=", opt.decay),
    ]
    print(io, "PlainGraves(", join(p, ", "), ")")
end

function update!(opt::PlainGraves)
    lr = opt.learning_rate
    momentum = opt.momentum
    decay = opt.decay
    umdecay = 1 - decay
    epsilon = opt.epsilon
    for k in eachindex(opt.paramvec)
        param = opt.paramvec[k]
        cache = opt.cachevec[k]
        m1 = opt.m1vec[k]
        m2 = opt.m2vec[k]
        @assert size(param) == size(m1) == size(m2)
        @flimsy_inbounds for i in eachindex(param)
            g = param.grad[i]
            g2 = g * g
            m1[i] = decay * m1[i] + umdecay * g
            m2[i] = decay * m2[i] + umdecay * g2
            cache[i] = momentum * cache[i] - lr * g / sqrt(m2[i] - m1[i] * m1[i] + epsilon)
            param.data[i] += cache[i]
        end
        fill!(param.grad, 0)
    end
end


####################
# --- AdaDelta --- #
####################
abstract AdaDelta <: Optimizer

# Plain
type PlainAdaDelta{T<:AbstractFloat} <: AdaDelta
    decay::T
    paramvec::Vector{Variable}
    cachevec::Vector{Matrix{FloatX}}
    deltavec::Vector{Matrix{FloatX}}
end

function Base.show(io::IO, opt::PlainAdaDelta)
    p = [
        string("decay=", opt.decay),
    ]
    print(io, "PlainAdaDelta(", join(p, ", "), ")")
end

function update!(opt::AdaDelta)
    decay = opt.decay
    umdecay = 1 - decay
    for i in eachindex(opt.paramvec)
        param = opt.paramvec[i]
        g2_tm1 = opt.cachevec[i]
        d2_tm1 = opt.deltavec[i]
        @assert size(param) == size(g2_tm1) == size(d2_tm1)
        @flimsy_inbounds for i in eachindex(param)
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

# Scale
type ScaledAdaDelta{T<:AbstractFloat} <: AdaDelta
    decay::T
    max_norm::T
    paramvec::Vector{Variable}
    cachevec::Vector{Matrix{FloatX}}
    deltavec::Vector{Matrix{FloatX}}
end

function Base.show(io::IO, opt::ScaledAdaDelta)
    p = [
        string("decay=", opt.decay),
        string("max_norm=", opt.max_norm),

    ]
    print(io, "ScaledAdaDelta(", join(p, ", "), ")")
end

function update!(opt::ScaledAdaDelta)
    decay = opt.decay
    umdecay = 1 - decay
    gnorm = norm_param_grad(opt.paramvec)
    scale = gnorm > opt.max_norm ? opt.max_norm / gnorm : 1.0

    for i in eachindex(opt.paramvec)
        param = opt.paramvec[i]
        g2_tm1 = opt.cachevec[i]
        d2_tm1 = opt.deltavec[i]
        @assert size(param) == size(g2_tm1) == size(d2_tm1)
        @flimsy_inbounds for i in eachindex(param)
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

# Clip
type ClippedAdaDelta{T<:AbstractFloat} <: AdaDelta
    decay::T
    clip::T
    paramvec::Vector{Variable}
    cachevec::Vector{Matrix{FloatX}}
    deltavec::Vector{Matrix{FloatX}}
end

function Base.show(io::IO, opt::ClippedAdaDelta)
    p = [
        string("decay=", opt.decay),
        string("clip=", opt.clip),

    ]
    print(io, "ClippedAdaDelta(", join(p, ", "), ")")
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
        @flimsy_inbounds for i in eachindex(param)
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

# Plain
type PlainAdam{T<:AbstractFloat} <: Adam
    learning_rate::T
    moment1_decay::T
    moment2_decay::T
    epsilon::T
    timestep::Int
    paramvec::Vector{Variable}
    moment1_vec::Vector{Matrix{FloatX}}
    moment2_vec::Vector{Matrix{FloatX}}
end

function Base.show(io::IO, opt::PlainAdam)
    p = [
        string("learning_rate=", opt.learning_rate),
        string("m1=", opt.moment1_decay),
        string("m2=", opt.moment2_decay),
        string("epsilon=", opt.epsilon),
        string("timestep=", opt.timestep),

    ]
    print(io, "PlainAdam(", join(p, ", "), ")")
end

function update!(opt::Adam, increment::Bool=false)
    if increment
        opt.timestep += 1
    end
    lr = opt.learning_rate
    b1 = opt.moment1_decay
    b2 = opt.moment2_decay
    e = opt.epsilon
    t = opt.timestep
    umb1 = 1 - b1
    umb2 = 1 - b2
    umb1sqr = 1 - (b1 * b1)^t
    umb2sqr = 1 - (b2 * b2)^t

    for i in eachindex(opt.paramvec)
        param = opt.paramvec[i]
        m1 = opt.moment1_vec[i]
        m2 = opt.moment2_vec[i]
        @assert size(param) == size(m1) == size(m2)
        @flimsy_inbounds for i in eachindex(param)
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

# Scale
type ScaledAdam{T<:AbstractFloat} <: Adam
    learning_rate::T
    moment1_decay::T
    moment2_decay::T
    epsilon::T
    max_norm::T
    timestep::Int
    paramvec::Vector{Variable}
    moment1_vec::Vector{Matrix{FloatX}}
    moment2_vec::Vector{Matrix{FloatX}}
end

function Base.show(io::IO, opt::ScaledAdam)
    p = [
        string("learning_rate=", opt.learning_rate),
        string("m1=", opt.moment1_decay),
        string("m2=", opt.moment2_decay),
        string("epsilon=", opt.epsilon),
        string("max_norm=", opt.max_norm),
        string("timestep=", opt.timestep),

    ]
    print(io, "ScaledAdam(", join(p, ", "), ")")
end

function update!(opt::ScaledAdam, increment::Bool=false)
    if increment
        opt.timestep += 1
    end
    lr = opt.learning_rate
    b1 = opt.moment1_decay
    b2 = opt.moment2_decay
    e = opt.epsilon
    t = opt.timestep
    umb1 = 1 - b1
    umb2 = 1 - b2
    umb1sqr = 1 - (b1 * b1)^t
    umb2sqr = 1 - (b2 * b2)^t
    gnorm = norm_param_grad(opt.paramvec)
    scale = gnorm > opt.max_norm ? opt.max_norm / gnorm : 1.0

    for i in eachindex(opt.paramvec)
        param = opt.paramvec[i]
        m1 = opt.moment1_vec[i]
        m2 = opt.moment2_vec[i]
        @assert size(param) == size(m1) == size(m2)
        @flimsy_inbounds for i in eachindex(param)
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

# Clip
type ClippedAdam{T<:AbstractFloat} <: Adam
    learning_rate::T
    moment1_decay::T
    moment2_decay::T
    epsilon::T
    clip::T
    timestep::Int
    paramvec::Vector{Variable}
    moment1_vec::Vector{Matrix{FloatX}}
    moment2_vec::Vector{Matrix{FloatX}}
end

function Base.show(io::IO, opt::ClippedAdam)
    p = [
        string("learning_rate=", opt.learning_rate),
        string("m1=", opt.moment1_decay),
        string("m2=", opt.moment2_decay),
        string("epsilon=", opt.epsilon),
        string("clip=", opt.clip),
        string("timestep=", opt.timestep),

    ]
    print(io, "ClippedAdam(", join(p, ", "), ")")
end

function update!(opt::ClippedAdaDelta, increment::Bool=false)
    if increment
        opt.timestep += 1
    end
    lr = opt.learning_rate
    b1 = opt.moment1_decay
    b2 = opt.moment2_decay
    e = opt.epsilon
    t = opt.timestep
    umb1 = 1 - b1
    umb2 = 1 - b2
    umb1sqr = 1 - (b1 * b1)^t
    umb2sqr = 1 - (b2 * b2)^t
    ub = opt.clip
    lb = -opt.clip

    for i in eachindex(opt.paramvec)
        param = opt.paramvec[i]
        m1 = opt.moment1_vec[i]
        m2 = opt.moment2_vec[i]
        @assert size(param) == size(m1) == size(m2)
        @flimsy_inbounds for i in eachindex(param)
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
    clipping_type::Union{Symbol,AbstractString}=:none,
    clip::Real=5.0,
    decay::Real=0.9,
    moment1_decay::Real=0.9,
    moment2_decay::Real=0.999,
    epsilon::Real=1e-9,
    momentum::Real=0.87,
    )

    if !isa(clipping_type, Symbol)
        clipping_type = Symbol(clipping_type)
    end

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
    elseif O <: RmsProp
        paramvec = convert(Vector, theta)
        cachevec = Cache(paramvec)
        if clipping_type == :none
            return PlainRmsProp(learning_rate, decay, paramvec, cachevec)
        elseif clipping_type == :scale
            return ScaledRmsProp(learning_rate, decay, clip, paramvec, cachevec)
        elseif clipping_type == :clip
            return ClippedRmsProp(learning_rate, decay, clip, paramvec, cachevec)
        else
            error("unkown clipping_type: $clipping_type for optimizer: $O")
        end
    elseif O <: Graves
        paramvec = convert(Vector, theta)
        cachevec = Cache(paramvec)
        m1vec = Cache(paramvec)
        m2vec = Cache(paramvec)
        return PlainGraves(learning_rate, momentum, decay, 0.0001, paramvec, cachevec, m1vec, m2vec)
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
            return PlainAdam(learning_rate, moment1_decay, moment2_decay, epsilon, 1, paramvec, moment1_vec, moment2_vec)
        elseif clipping_type == :scale
            return ScaledAdam(learning_rate, moment1_decay, moment2_decay, epsilon, clip, 1, paramvec, moment1_vec, moment2_vec)
        elseif clipping_type == :clip
            return ClippedAdam(learning_rate, moment1_decay, moment2_decay, epsilon, clip, 1, paramvec, moment1_vec, moment2_vec)
        else
            error("unkown clipping_type: $clipping_type for optimizer: $O")
        end
    else
        error("uknown optimizer: $O")
    end
end
