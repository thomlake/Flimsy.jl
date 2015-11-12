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
function cat{T,M}(target::Integer, output::Variable{T,M,1}, eps::Float64=1e-20)
    pr_target = output.data[target] + eps
    nll = -log(pr_target)
    isfinite(nll) || error("nll: $nll not finite")
    return nll
end

function cat{T,M}(stack::BPStack, target::Integer, output::Variable{T,M,1}, eps::Float64=1e-20)
    pr_target = output.data[target] + eps
    output.grad[target] -= 1 / pr_target
    nll = -log(pr_target)
    isfinite(nll) || error("nll: $nll not finite")
    return nll
end

function cat{I<:Integer}(target::Vector{I}, output::Variable, eps::Float64=1e-20)
    @assert size(output, 2) == length(target)

    nll = 0.0
    for i = 1:size(output, 2)
        pr_target = output.data[target[i],i] + eps
        nll -= log(pr_target)
    end
    isfinite(nll) || error("nll: $nll not finite")
    return nll
end

function cat{I<:Integer}(stack::BPStack, target::Vector{I}, output::Variable, eps::Float64=1e-20)
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
