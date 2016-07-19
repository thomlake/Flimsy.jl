"""
**Categorical Cross Entropy With Scores**

Compute categorical cross entropy from scores by first normalizing.
Equivalent to `categorical_cross_entropy(softmax(scores))` but more efficient.
Categorical cross entropy is equivalent to the negative log likelihood of a categorical random variable.

`categorical_cross_entropy_with_scores(output::Variable, target::Union{Integer, Vector{Integer}})`

`categorical_cross_entropy_with_scores(output::Variable, target::Union{Integer, Vector{Integer}}, weight::Real)`
"""
function categorical_cross_entropy_with_scores end

# ----------- #
# Mx1 Integer #
# ----------- #
function categorical_cross_entropy_with_scores(scope::Scope, output::AbstractValue, target::Integer)
    size(output, 2) == 1 || throw(DimensionMismatch("output must be size Mx1 for Integer target but got $(size(output))"))
    pr = softmax(output.data)
    pr_target = pr[target] + CROSS_ENTROPY_EPS
    nll = -log(pr_target)
    isfinite(nll) || throw(ErrorException("nll not finite ($nll)"))
    return nll
end

function categorical_cross_entropy_with_scores(scope::GradScope, output::Variable, target::Integer)
    size(output, 2) == 1 || throw(DimensionMismatch("output must be size Mx1 for Integer target but got $(size(output))"))
    pr = softmax(output.data)
    pr_target = pr[target] + CROSS_ENTROPY_EPS
    for i in eachindex(pr)
        output.grad[i] += pr[i] - (i == target ? 1 : 0)
    end
    nll = -log(pr_target)
    isfinite(nll) || throw(ErrorException("nll not finite ($nll)"))
    return nll
end

function categorical_cross_entropy_with_scores(scope::Scope, output::AbstractValue, target::Integer, weight::Real)
    return weight * categorical_cross_entropy_with_scores(scope, output, target)
end

function categorical_cross_entropy_with_scores(scope::GradScope, output::Variable, target::Integer, weight::Real)
    size(output, 2) == 1 || throw(DimensionMismatch("output must be size Mx1 for Integer target but got $(size(output))"))
    pr = softmax(output.data)
    pr_target = pr[target] + CROSS_ENTROPY_EPS
    for i in eachindex(pr)
        output.grad[i] += weight * (pr[i] - (i == target ? 1 : 0))
    end
    nll = -log(pr_target)
    isfinite(nll) || throw(ErrorException("nll not finite ($nll)"))
    return weight * nll
end

# ------------------- #
# MxN Vector{Integer} #
# ------------------- #
function categorical_cross_entropy_with_scores{I<:Integer}(scope::Scope, output::AbstractValue, target::Vector{I})
    size(output, 2) == length(target) || throw(DimensionMismatch("output must be size Mx$(length(target)) for Integer target vector with length $(length(target)) but got $(size(output))"))
    pr = softmax(output.data)
    nll = 0.0
    for j = 1:length(target)
        nll -= log(pr[target[j],j] + CROSS_ENTROPY_EPS)
    end
    isfinite(nll) || throw(ErrorException("nll not finite ($nll)"))
    return nll
end

function categorical_cross_entropy_with_scores{I<:Integer}(scope::GradScope, output::Variable, target::Vector{I})
    size(output, 2) == length(target) || throw(DimensionMismatch("output must be size Mx$(length(target)) for Integer target vector with length $(length(target)) but got $(size(output))"))
    pr = softmax(output.data)
    nll = 0.0
    for j = 1:size(pr, 2)
        for i = 1:size(pr, 1)
            output.grad[i,j] += pr[i,j] - (i == target[j] ? 1 : 0)
        end
        nll -= log(pr[target[j], j] + CROSS_ENTROPY_EPS)
    end
    isfinite(nll) || throw(ErrorException("nll not finite ($nll)"))
    return nll
end

function categorical_cross_entropy_with_scores{I<:Integer}(scope::Scope, output::AbstractValue, target::Vector{I}, weight::Real)
    return weight * categorical_cross_entropy_with_scores(scope, output, target)
end

function categorical_cross_entropy_with_scores{I<:Integer}(scope::GradScope, output::Variable, target::Vector{I}, weight::Real)
    size(output, 2) == length(target) || throw(DimensionMismatch("output must be size Mx$(length(target)) for Integer target vector with length $(length(target)) but got $(size(output))"))
    pr = softmax(output.data)
    nll = 0.0
    for j = 1:size(pr, 2)
        for i = 1:size(pr, 1)
            output.grad[i,j] += weight * (pr[i,j] - (i == target[j] ? 1 : 0))
        end
        nll -= log(pr[target[j], j] + CROSS_ENTROPY_EPS)
    end
    isfinite(nll) || throw(ErrorException("nll not finite ($nll)"))
    return weight * nll
end
