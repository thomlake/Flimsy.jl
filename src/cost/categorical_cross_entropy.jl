"""Categorical Cross Entropy, i.e., negative log likelihood of a categorical random variable.
"""
function categorical_cross_entropy end

function categorical_cross_entropy(output::Variable, target::Integer, eps::AbstractFloat=1e-20)
    size(output, 2) == 1 || throw(DimensionMismatch("output must be size Mx1 for Integer target"))
    pr_target = output.data[target] + eps
    nll = -log(pr_target)
    isfinite(nll) || throw(ErrorException("nll not finite ($nll)"))
    return nll
end

function categorical_cross_entropy(stack::CallbackStack, output::GradVariable, target::Integer, eps::AbstractFloat=1e-20)
    size(output, 2) == 1 || throw(DimensionMismatch("output must be size Mx1 for Integer target"))
    pr_target = output.data[target] + eps
    output.grad[target] -= 1 / pr_target
    nll = -log(pr_target)
    isfinite(nll) || throw(ErrorException("nll not finite ($nll)"))
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