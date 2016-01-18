"""Categorical Cross Entropy, i.e., negative log likelihood of a categorical random variable.
"""
function categorical_cross_entropy(y, target::Integer, eps::AbstractFloat=1e-20)
    sz = size(y)
    length(sz) == 1 || sz[2] == 1 || error("size mismatch")
    pr_target = y[target] + eps
    nll = -log(pr_target)
    isfinite(nll) || error("nll: $nll not finite")
    return nll
end

function categorical_cross_entropy(stack::CallbackStack, Y::GradPair, target::Integer, eps::AbstractFloat=1e-20)
    sz = size(Y)
    length(sz) == 1 || sz[2] == 1 || error("size mismatch")
    y, dy = Y
    pr_target = y[target] + eps
    dy[target] -= 1 / pr_target
    nll = -log(pr_target)
    isfinite(nll) || error("nll: $nll not finite")
    return nll
end

"""Categorical Cross Entropy from unnormalized scores.
"""
function categorical_cross_entropy_with_score(s, target::Integer, eps::AbstractFloat=1e-20)
    sz = size(s)
    length(sz) == 1 || sz[2] == 1 || error("size mismatch")
    y = softmax(s)
    pr_target = y[target] + eps
    nll = -log(pr_target)
    isfinite(nll) || error("nll: $nll not finite")
    return nll
end

function categorical_cross_entropy_with_score(stack::CallbackStack, S::GradPair, target::Integer, eps::AbstractFloat=1e-20)
    sz = size(S)
    length(sz) == 1 || sz[2] == 1 || error("size mismatch")
    s, ds = S
    y = softmax(s)
    for i in eachindex(y)
        ds[i] += (i == target ? 1 : 0) - y[i]
    end
    pr_target = y[target] + eps
    nll = -log(pr_target)
    isfinite(nll) || error("nll: $nll not finite")
    return nll
end

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