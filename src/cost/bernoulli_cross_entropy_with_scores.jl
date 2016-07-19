"""
**Bernoulli Cross Entropy With Scores**

Compute Bernoulli cross entropy from scores by first passing through sigmoid to variables are in range [0, 1].
Equivalent to `bernoulli_cross_entropy_with_scores(sigmoid(scores))` but more efficient.
Bernoulli cross entropy is equivalent to the negative joint log likelihood of independent Bernoulli random variables.

`bernoulli_cross_entropy_with_scores(output::Variable, target::Union{Bool, Vector{Bool}, Matrix{Bool}})`

`bernoulli_cross_entropy_with_scores(output::Variable, target::Union{Bool, Vector{Bool}, Matrix{Bool}}, weight::Real)`
"""

function bernoulli_cross_entropy_with_scores(scope::Scope, output::AbstractValue, target::AbstractMatrix{Bool})
    size(output) == size(target) || throw(DimensionMismatch("output and target do not have the same size"))
    pr = sigmoid(output.data)
    nll = 0.0
    for i in eachindex(target)
        pr_target = target[i] ? pr[i] : 1 - pr[i]
        pr_target = max(pr_target, CROSS_ENTROPY_EPS)
        nll -= log(pr_target)
    end
    isfinite(nll) || error("nll: $nll not finite")
    return nll
end

function bernoulli_cross_entropy_with_scores(scope::GradScope, output::Variable, target::AbstractMatrix{Bool})
    size(output) == size(target) || throw(DimensionMismatch("output and target do not have the same size"))
    pr = sigmoid(output.data)
    nll = 0.0
    pr_target = 0.0
    for i in eachindex(target)
        if target[i]
            pr_target = pr[i]
            output.grad[i] += pr[i] - 1
        else
            pr_target = 1 - pr[i]
            output.grad[i] += pr[i]
        end
        pr_target = max(pr_target, CROSS_ENTROPY_EPS)
        nll -= log(pr_target)
    end
    isfinite(nll) || error("nll: $nll not finite")
    return nll
end

function bernoulli_cross_entropy_with_scores(scope::Scope, output::AbstractValue, target::AbstractMatrix{Bool}, weight::Real)
    return weight * bernoulli_cross_entropy_with_scores(output, target)
end

function bernoulli_cross_entropy_with_scores(scope::GradScope, output::Variable, target::AbstractMatrix{Bool}, weight::Real)
    size(output) == size(target) || throw(DimensionMismatch("output and target do not have the same size"))
    pr = sigmoid(output.data)
    nll = 0.0
    pr_target = 0.0
    for i in eachindex(target)
        if target[i]
            pr_target = pr[i]
            output.grad[i] += weight * (pr[i] - 1)
        else
            pr_target = 1 - pr[i]
            output.grad[i] += weight * pr[i]
        end
        pr_target = max(pr_target, CROSS_ENTROPY_EPS)
        nll -= log(pr_target)
    end
    isfinite(nll) || error("nll: $nll not finite")
    return weight * nll
end

bernoulli_cross_entropy_with_scores(scope::Scope, o::AbstractValue, t::Bool) = 
    bernoulli_cross_entropy_with_scores(scope, o, fill(t, 1, 1))

bernoulli_cross_entropy_with_scores(scope::GradScope, o::AbstractValue, t::Bool) = 
    bernoulli_cross_entropy_with_scores(scope, o, fill(t, 1, 1))

bernoulli_cross_entropy_with_scores(scope::Scope, o::AbstractValue, t::Bool, w::Real) = 
    bernoulli_cross_entropy_with_scores(scope, o, fill(t, 1, 1), w)

bernoulli_cross_entropy_with_scores(scope::GradScope, o::AbstractValue, t::Bool, w::Real) = 
    bernoulli_cross_entropy_with_scores(scope, o, fill(t, 1, 1), w)

bernoulli_cross_entropy_with_scores(scope::Scope, o::AbstractValue, t::AbstractVector{Bool}) = 
    bernoulli_cross_entropy_with_scores(scope, o, reshape(t, length(t), 1))

bernoulli_cross_entropy_with_scores(scope::GradScope, o::AbstractValue, t::AbstractVector{Bool}) = 
    bernoulli_cross_entropy_with_scores(scope, o, reshape(t, length(t), 1))

bernoulli_cross_entropy_with_scores(scope::Scope, o::AbstractValue, t::AbstractVector{Bool}, w::Real) = 
    bernoulli_cross_entropy_with_scores(scope, o, reshape(t, length(t), 1), w)

bernoulli_cross_entropy_with_scores(scope::GradScope, o::AbstractValue, t::AbstractVector{Bool}, w::Real) = 
    bernoulli_cross_entropy_with_scores(scope, o, reshape(t, length(t), 1), w)
