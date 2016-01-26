"""
**Bernoulli Cross Entropy**

Each element of output should be in [0, 1].
Bernoulli cross entropy is equivalent to the negative joint log likelihood of independent Bernoulli random variables.

`bernoulli_cross_entropy(output::Variable, target::Union{Bool, Vector{Bool}, Matrix{Bool}})`

`bernoulli_cross_entropy(output::Variable, target::Union{Bool, Vector{Bool}, Matrix{Bool}}, weight::Real)`
"""

function bernoulli_cross_entropy(output::Variable, target::AbstractMatrix{Bool})
    size(output) == size(target) || throw(DimensionMismatch("output and target do not have the same size"))
    nll = 0.0
    for i in eachindex(target)
        pr_target = target[i] ? output.data[i] : 1 - output.data[i]
        pr_target = max(pr_target, CROSS_ENTROPY_EPS)
        nll -= log(pr_target)
    end
    isfinite(nll) || error("nll: $nll not finite")
    return nll
end

function bernoulli_cross_entropy(stack::CallbackStack, output::GradVariable, target::AbstractMatrix{Bool})
    size(output) == size(target) || throw(DimensionMismatch("output and target do not have the same size"))
    nll = 0.0
    pr_target = 0.0
    for i in eachindex(target)
        if target[i]
            pr_target = output.data[i]
            pr_target = max(pr_target, CROSS_ENTROPY_EPS)
            output.grad[i] -= 1 / pr_target
        else
            pr_target = 1 - output.data[i]
            pr_target = max(pr_target, CROSS_ENTROPY_EPS)
            output.grad[i] += 1 / pr_target
            p
        end
        nll -= log(pr_target)
    end
    isfinite(nll) || error("nll: $nll not finite")
    return nll
end

function bernoulli_cross_entropy(output::Variable, target::AbstractMatrix{Bool}, weight::Real)
    return weight * bernoulli_cross_entropy(output, target)
end

function bernoulli_cross_entropy(stack::CallbackStack, output::GradVariable, target::AbstractMatrix{Bool}, weight::Real)
    size(output) == size(target) || throw(DimensionMismatch("output and target do not have the same size"))
    nll = 0.0
    pr_target = 0.0
    for i in eachindex(target)
        if target[i]
            pr_target = output.data[i]
            pr_target = max(pr_target, CROSS_ENTROPY_EPS)
            output.grad[i] -= weight / pr_target
        else
            pr_target = 1 - output.data[i]
            pr_target = max(pr_target, CROSS_ENTROPY_EPS)
            output.grad[i] += weight / pr_target
            p
        end
        nll -= log(pr_target)
    end
    isfinite(nll) || error("nll: $nll not finite")
    return weight * nll
end

bernoulli_cross_entropy(o::Variable, t::Bool) = 
    bernoulli_cross_entropy(o, fill(t, 1, 1))

bernoulli_cross_entropy(s::CallbackStack, o::Variable, t::Bool) = 
    bernoulli_cross_entropy(s, o, fill(t, 1, 1))

bernoulli_cross_entropy(o::Variable, t::Bool, w::Real) = 
    bernoulli_cross_entropy(o, fill(t, 1, 1), w)

bernoulli_cross_entropy(s::CallbackStack, o::Variable, t::Bool, w::Real) = 
    bernoulli_cross_entropy(s, o, fill(t, 1, 1), w)

bernoulli_cross_entropy(o::Variable, t::AbstractVector{Bool}) = 
    bernoulli_cross_entropy(o, reshape(t, length(t), 1))

bernoulli_cross_entropy(s::CallbackStack, o::Variable, t::AbstractVector{Bool}) = 
    bernoulli_cross_entropy(s, o, reshape(t, length(t), 1))

bernoulli_cross_entropy(o::Variable, t::AbstractVector{Bool}, w::Real) = 
    bernoulli_cross_entropy(o, reshape(t, length(t), 1), w)

bernoulli_cross_entropy(s::CallbackStack, o::Variable, t::AbstractVector{Bool}, w::Real) = 
    bernoulli_cross_entropy(s, o, reshape(t, length(t), 1), w)
