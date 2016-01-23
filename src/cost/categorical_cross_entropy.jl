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


function categorical_cross_entropy{I<:Integer}(output::Variable, target::Vector{I}, eps::AbstractFloat=1e-20)
    n = length(target)
    size(output, 2) == n || throw(DimensionMismatch("output must be size Mx$n"))

    nll = 0.0
    for j = 1:n
        pr_target = output.data[target[j],j] + eps
        nll -= log(pr_target)
    end
    isfinite(nll) || error("nll: $nll not finite")
    return nll
end

function categorical_cross_entropy{I<:Integer}(stack::CallbackStack, output::GradVariable, target::Vector{I}, eps::AbstractFloat=1e-20)
    n = length(target)
    size(output, 2) == n || throw(DimensionMismatch("output must be size Mx$n"))

    nll = 0.0
    for j = 1:n
        pr_target = output.data[target[j],j] + eps
        output.grad[target[j],j] -= 1 / pr_target
        nll -= log(pr_target)
    end
    isfinite(nll) || error("nll: $nll not finite")
    return nll
end
