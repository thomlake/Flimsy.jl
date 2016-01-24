"""
**Categorical Cross Entropy**

If `target` is an integer `output` should be a vector which sums to 1.
If `target` is a vector of integers `output` should be a matrix with columns that sum to 1.
Sum contraints are *not* checked at runtime.
Categorical cross entropy is equivalent to the negative log likelihood of a categorical random variable.

`categorical_cross_entropy(output::Variable, target::Union{Integer, Vector{Integer}})`

`categorical_cross_entropy(output::Variable, target::Union{Integer, Vector{Integer}}, weight::Real)`
"""
function categorical_cross_entropy end

# ----------- #
# Mx1 Integer #
# ----------- #
function categorical_cross_entropy(output::Variable, target::Integer)
    size(output, 2) == 1 || throw(DimensionMismatch("output must be size Mx1 for Integer target"))
    pr_target = output.data[target] + CROSS_ENTROPY_EPS
    nll = -log(pr_target)
    isfinite(nll) || throw(ErrorException("nll not finite ($nll)"))
    return nll
end

function categorical_cross_entropy(stack::CallbackStack, output::GradVariable, target::Integer)
    size(output, 2) == 1 || throw(DimensionMismatch("output must be size Mx1 for Integer target"))
    pr_target = output.data[target] + CROSS_ENTROPY_EPS
    output.grad[target] -= 1 / pr_target
    nll = -log(pr_target)
    isfinite(nll) || throw(ErrorException("nll not finite ($nll)"))
    return nll
end

function categorical_cross_entropy(output::Variable, target::Integer, weight::Real)
    return weight * categorical_cross_entropy(output, target)
end

function categorical_cross_entropy(stack::CallbackStack, output::GradVariable, target::Integer, weight::Real)
    size(output, 2) == 1 || throw(DimensionMismatch("output must be size Mx1 for Integer target"))
    pr_target = output.data[target] + CROSS_ENTROPY_EPS
    output.grad[target] -= weight / pr_target
    nll = -log(pr_target)
    isfinite(nll) || throw(ErrorException("nll not finite ($nll)"))
    return weight * nll
end

# ------------------- #
# MxN Vector{Integer} #
# ------------------- #
function categorical_cross_entropy{I<:Integer}(output::Variable, target::Vector{I})
    n = length(target)
    size(output, 2) == n || throw(DimensionMismatch("output must be size Mx$n"))
    nll = 0.0
    for j = 1:n
        pr_target = output.data[target[j],j] + CROSS_ENTROPY_EPS
        nll -= log(pr_target)
    end
    isfinite(nll) || error("nll: $nll not finite")
    return nll
end

function categorical_cross_entropy{I<:Integer}(stack::CallbackStack, output::GradVariable, target::Vector{I})
    n = length(target)
    size(output, 2) == n || throw(DimensionMismatch("output must be size Mx$n"))
    nll = 0.0
    for j = 1:n
        pr_target = output.data[target[j],j] + CROSS_ENTROPY_EPS
        output.grad[target[j],j] -= 1 / pr_target
        nll -= log(pr_target)
    end
    isfinite(nll) || error("nll: $nll not finite")
    return nll
end

function categorical_cross_entropy{I<:Integer}(output::Variable, target::Vector{I}, weight::Real)
    return weight * categorical_cross_entropy(output, target)
end

function categorical_cross_entropy{I<:Integer}(stack::CallbackStack, output::GradVariable, target::Vector{I}, weight::Real)
    n = length(target)
    size(output, 2) == n || throw(DimensionMismatch("output must be size Mx$n"))
    nll = 0.0
    for j = 1:n
        pr_target = output.data[target[j],j] + CROSS_ENTROPY_EPS
        output.grad[target[j],j] -= weight / pr_target
        nll -= log(pr_target)
    end
    isfinite(nll) || error("nll: $nll not finite")
    return weight * nll
end
