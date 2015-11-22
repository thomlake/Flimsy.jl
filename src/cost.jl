# Gaussian negative log likelihood loss function
function gauss{T}(target::AbstractFloat, output::Variable{T,1,1})
    delta = output.data[1] - target
    return 0.5 * delta * delta
end

function gauss{T}(stack::BPStack, target::AbstractFloat, output::Variable{T,1,1})
    delta = output.data[1] - target
    output.grad[1] += delta
    return 0.5 * delta * delta
end

function gauss{T<:AbstractFloat}(target::Array{T}, output::Variable)
    @assert size(target) == size(output)

    sse = 0.0
    for i = 1:endof(target)
        delta = output.data[i] - target[i]
        sse += delta * delta
    end
    return 0.5 * sse
end


function gauss{T<:AbstractFloat}(stack::BPStack, target::Array{T}, output::Variable)
    @assert size(target) == size(output)

    sse = 0.0
    for i = 1:endof(target)
        delta = output.data[i] - target[i]
        output.grad[i] += delta
        sse += delta * delta
    end
    return 0.5 * sse
end

# Categorical negative log likelihood loss function
function cat{T,M}(target::Integer, output::Variable{T,M,1}, eps::AbstractFloat=1e-20)
    pr_target = output.data[target] + eps
    nll = -log(pr_target)
    isfinite(nll) || error("nll: $nll not finite")
    return nll
end

function cat{T,M}(stack::BPStack, target::Integer, output::Variable{T,M,1}, eps::AbstractFloat=1e-20)
    pr_target = output.data[target] + eps
    output.grad[target] -= 1 / pr_target
    nll = -log(pr_target)
    isfinite(nll) || error("nll: $nll not finite")
    return nll
end

function cat{I<:Integer}(target::Vector{I}, output::Variable, eps::AbstractFloat=1e-20)
    @assert size(output, 2) == length(target)

    nll = 0.0
    for i = 1:size(output, 2)
        pr_target = output.data[target[i],i] + eps
        nll -= log(pr_target)
    end
    isfinite(nll) || error("nll: $nll not finite")
    return nll
end

function cat{I<:Integer}(stack::BPStack, target::Vector{I}, output::Variable, eps::AbstractFloat=1e-20)
    @assert size(output, 2) == length(target)

    nll = 0.0
    for i = 1:size(output, 2)
        pr_target = output.data[target[i],i] + eps
        output.grad[target[i],i] -= 1 / pr_target
        nll -= log(pr_target)
    end
    isfinite(nll) || error("nll: $nll not finite")
    return nll
end

# Bernoulli negative log likelihood loss function
# function bern{T}(target::Bool, output::Variable{T,1,1}, eps::AbstractFloat=1e-20)
#     pr_target = target ? output.data[1] : 1 - output.data[1]
#     pr_target = max(eps, pr_target)
#     nll = -log(pr_target)
#     isfinite(nll) || error("nll: $nll not finite")
#     return nll
# end

# function bern{T}(stack::BPStack, target::Bool, output::Variable{T,1,1}, eps::AbstractFloat=1e-20)

#     pr_target = if target 
#         p = output.data[1]
#         p = max(eps, p)
#         output.grad[1] -= 1 / p
#         p
#     else
#         p = 1 - output.data[1]
#         p = max(eps, p)
#         output.grad[1] += 1 / p
#         p
#     end

#     nll = -log(pr_target)
#     isfinite(nll) || error("nll: $nll not finite")
#     return nll
# end

# bern{T}(target::Vector{Bool}, output::Variable{T,1,1}, eps::AbstractFloat=1e-20) = length(target) == 1 ? bern(target[1], output, eps) : error("not enough outputs")

# bern{T}(stack::BPStack, target::Vector{Bool}, output::Variable{T,1,1}, eps::AbstractFloat=1e-20) = length(target) == 1 ? bern(stack, target[1], output, eps) : error("not enough outputs")

# function bern{T,N}(target::Vector{Bool}, output::Variable{T,1,N}, eps::AbstractFloat=1e-20)
#     @assert length(target) == N

#     nll = 0.0
#     for i = 1:N
#         pr_target = target[i] ? output.data[i] : 1 - output.data[i]
#         pr_target = max(eps, pr_target)
#         nll -= log(pr_target)
#     end
#     isfinite(nll) || error("nll: $nll not finite")
#     return nll
# end

# function bern{T,N}(stack::BPStack, target::Vector{Bool}, output::Variable{T,1,N}, eps::AbstractFloat=1e-20)
#     @assert length(target) == N
#     nll = 0.0
#     for i = 1:N
#         pr_target = if target[i]
#             p = output.data[i]
#             p = max(eps, p)
#             output.grad[i] -= 1 / p
#             p
#         else
#             p = 1 - output.data[i]
#             p = max(eps, p)
#             output.grad[i] += 1 / p
#             p
#         end
#         nll -= log(pr_target)
#     end
#     isfinite(nll) || error("nll: $nll not finite")
#     return nll
# end

# # Multilabel Bernoulli negative log likelihood loss function
# function bern{T,M}(target::Vector{Bool}, output::Variable{T,M,1}, eps::AbstractFloat=1e-20)
#     @assert length(target) == M

#     nll = 0.0
#     for i = 1:M
#         pr_target = target[i] ? output.data[i] : 1 - output.data[i]
#         pr_target = max(eps, pr_target)
#         nll -= log(pr_target)
#     end
#     isfinite(nll) || error("nll: $nll not finite")
#     return nll
# end

# function bern{T,M}(stack::BPStack, target::Vector{Bool}, output::Variable{T,M,1}, eps::AbstractFloat=1e-20)
#     @assert length(target) == M
#     nll = 0.0
#     for i = 1:M
#         pr_target = if target[i]
#             p = output.data[i]
#             p = max(eps, p)
#             output.grad[i] -= 1 / p
#             p
#         else
#             p = 1 - output.data[i]
#             p = max(eps, p)
#             output.grad[i] += 1 / p
#             p
#         end
#         nll -= log(pr_target)
#     end
#     isfinite(nll) || error("nll: $nll not finite")
#     return nll
# end

function bern{T,M,N}(target::AbstractArray{Bool}, output::Variable{T,M,N}, eps::AbstractFloat=1e-20)
    @assert size(target) == (M, N)

    nll = 0.0
    for i in eachindex(target)
        pr_target = target[i] ? output.data[i] : 1 - output.data[i]
        pr_target = max(eps, pr_target)
        nll -= log(pr_target)
    end
    isfinite(nll) || error("nll: $nll not finite")
    return nll
end

function bern{T,M,N}(stack::BPStack, target::AbstractArray{Bool}, output::Variable{T,M,N}, eps::AbstractFloat=1e-20)
    @assert size(target) == (M, N)
    nll = 0.0
    for i in eachindex(target)
        pr_target = if target[i]
            p = output.data[i]
            p = max(eps, p)
            output.grad[i] -= 1 / p
            p
        else
            p = 1 - output.data[i]
            p = max(eps, p)
            output.grad[i] += 1 / p
            p
        end
        nll -= log(pr_target)
    end
    isfinite(nll) || error("nll: $nll not finite")
    return nll
end

bern{T}(target::Bool, output::Variable{T,1,1}, eps=AbstractFloat=1e-20) = bern(fill(target, 1, 1), output, eps)
bern{T}(stack::BPStack, target::Bool, output::Variable{T,1,1}, eps=AbstractFloat=1e-20) = bern(stack, fill(target, 1, 1), output, eps)

bern{T}(target::AbstractVector{Bool}, output::Variable{T,1,1}, eps=AbstractFloat=1e-20) = bern(reshape(target, 1, 1), output, eps)
bern{T}(stack::BPStack, target::AbstractVector{Bool}, output::Variable{T,1,1}, eps=AbstractFloat=1e-20) = bern(stack, reshape(target, 1, 1), output, eps)

bern{T,M}(target::AbstractVector{Bool}, output::Variable{T,M,1}, eps=AbstractFloat=1e-20) = bern(reshape(target, M, 1), output, eps)
bern{T,M}(stack::BPStack, target::AbstractVector{Bool}, output::Variable{T,M,1}, eps=AbstractFloat=1e-20) = bern(stack, reshape(target, M, 1), output, eps)

bern{T,N}(target::AbstractVector{Bool}, output::Variable{T,1,N}, eps=AbstractFloat=1e-20) = bern(reshape(target, 1, N), output, eps)
bern{T,N}(stack::BPStack, target::AbstractVector{Bool}, output::Variable{T,1,N}, eps=AbstractFloat=1e-20) = bern(stack, reshape(target, 1, N), output, eps)

# function bern{T,M,N}(target::Vector{Bool}, output::Variable{T,M,N}, eps::AbstractFloat=1e-20)
#     @assert length(target) == N

#     nll = 0.0
#     for i = 1:N
#         pr_target = target[i] ? output.data[i] : 1 - output.data[i]
#         pr_target = max(eps, pr_target)
#         nll -= log(pr_target)
#     end
#     isfinite(nll) || error("nll: $nll not finite")
#     return nll
# end

# function bern{T,M,N}(stack::BPStack, target::Vector{Bool}, output::Variable{T,M,N}, eps::AbstractFloat=1e-20)
#     @assert length(target) == N
#     nll = 0.0
#     for i = 1:N
#         pr_target = if target[i]
#             p = output.data[i]
#             p = max(eps, p)
#             output.grad[i] -= 1 / p
#             p
#         else
#             p = 1 - output.data[i]
#             p = max(eps, p)
#             output.grad[i] += 1 / p
#             p
#         end
#         nll -= log(pr_target)
#     end
#     isfinite(nll) || error("nll: $nll not finite")
#     return nll
# end

# function cat{T,M}(target::Integer, output::Variable{T,M,1}, eps::AbstractFloat=1e-20)
#     pr_target = output.data[target] + eps
#     nll = -log(pr_target)
#     isfinite(nll) || error("nll: $nll not finite")
#     return nll
# end

# function cat{T,M}(stack::BPStack, target::Integer, output::Variable{T,M,1}, eps::AbstractFloat=1e-20)
#     pr_target = output.data[target] + eps
#     output.grad[target] -= 1 / pr_target
#     nll = -log(pr_target)
#     isfinite(nll) || error("nll: $nll not finite")
#     return nll
# end

# function cat{I<:Integer}(target::Vector{I}, output::Variable, eps::AbstractFloat=1e-20)
#     @assert size(output, 2) == length(target)

#     nll = 0.0
#     for i = 1:size(output, 2)
#         pr_target = output.data[target[i],i] + eps
#         nll -= log(pr_target)
#     end
#     isfinite(nll) || error("nll: $nll not finite")
#     return nll
# end

# function cat{I<:Integer}(stack::BPStack, target::Vector{I}, output::Variable, eps::AbstractFloat=1e-20)
#     @assert size(output, 2) == length(target)

#     nll = 0.0
#     for i = 1:size(output, 2)
#         pr_target = output.data[target[i],i] + eps
#         output.grad[target[i],i] -= 1 / pr_target
#         nll -= log(pr_target)
#     end
#     isfinite(nll) || error("nll: $nll not finite")
#     return nll
# end

# Max Margin loss
function margin{T,M}(target::Integer, output::Variable{T,M,1}, m::Real=1)
    k = Flimsy.Extras.argmaxneq(output, [target])
    err = max(0, output.data[k] + m - output.data[target])
    return err
end

function margin{T,M}(stack::BPStack, target::Integer, output::Variable{T,M,1}, m::Real=1)
    k = Flimsy.Extras.argmaxneq(output, [target])
    err = max(0, output.data[k] + m - output.data[target])
    if err > 0
        output.grad[k] += 1
        output.grad[target] -= 1
    end 
    return err
end

function margin{I<:Integer}(target::Vector{I}, output::Variable, m::Real=1)
    @assert size(output, 2) == length(target)
    
    errsum = 0.0
    ks = Flimsy.Extras.argmaxneq(output, target)
    for i = 1:size(output, 2)
        errsum += max(0, output.data[ks[i], i] + m - output.data[target[i], i])
    end
    return errsum
end

function margin{I<:Integer}(stack::BPStack, target::Vector{I}, output::Variable, m::Real=1)
    @assert size(output, 2) == length(target)

    errsum = 0.0
    ks = Flimsy.Extras.argmaxneq(output, target)
    for i = 1:size(output, 2)
        err = max(0, output.data[ks[i], i] + m - output.data[target[i], i])
        if err > 0
            output.grad[ks[i], i] += 1
            output.grad[target[i], i] -= 1
            errsum += err
        end 
    end
    return errsum
end


# Connectionist Temporal Classification loss
function ctc{I<:Int,V<:Variable}(target::Vector{I}, output::Vector{V}, blank::Int)
    ys = CTC.expand(target, blank)
    T = length(output)
    S = size(output[1], 1)
    lpmat = CTC.make_lpmat(output)
    fmat = CTC.forward(ys, lpmat, blank)
    ll = Flimsy.Extras.logsumexp(fmat[end, end], fmat[end-1, end])
    return -ll
end

function ctc{I<:Integer,V<:Variable}(stack::BPStack, target::Vector{I}, output::Vector{V}, blank::Int)
    ys = CTC.expand(target, blank)
    T = length(output)
    S = size(output[1], 1)
    lpmat = CTC.make_lpmat(output)
    fmat = CTC.forward(ys, lpmat, blank)
    bmat = CTC.backward(ys, lpmat, blank)
    fbmat = fmat + bmat
    ll = Flimsy.Extras.logsumexp(fmat[end, end], fmat[end-1, end])
    p = exp(ll)

    for t = 1:T
        for k = 1:S
            total = -Inf
            for s in findin(ys, k)
                total = Flimsy.Extras.logsumexp(total, fbmat[s, t])
            end
            output[t].grad[k] = exp(lpmat[k,t]) - exp(total - ll)
        end
    end
    return -ll
end
