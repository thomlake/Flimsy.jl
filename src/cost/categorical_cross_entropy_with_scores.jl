"""Categorical Cross Entropy from unnormalized scores.
"""
function categorical_cross_entropy_with_scores end

function categorical_cross_entropy_with_scores(output::Variable, target::Integer; avg::Bool=true, eps::AbstractFloat=1e-20)
    size(output, 2) == 1 || throw(DimensionMismatch("output must be size Mx1 for Integer target but got $(size(output))"))
    pr = softmax(output.data)
    pr_target = pr[target] + eps
    nll = -log(pr_target)
    isfinite(nll) || throw(ErrorException("nll not finite ($nll)"))
    return nll
end

function categorical_cross_entropy_with_scores(stack::CallbackStack, output::GradVariable, target::Integer; avg::Bool=true, eps::AbstractFloat=1e-20)
    size(output, 2) == 1 || throw(DimensionMismatch("output must be size Mx1 for Integer target but got $(size(output))"))
    pr = softmax(output.data)
    pr_target = pr[target] + eps
    for i in eachindex(pr)
        output.grad[i] += pr[i] - (i == target ? 1 : 0)
    end
    nll = -log(pr_target)
    isfinite(nll) || throw(ErrorException("nll not finite ($nll)"))
    return nll
end

function categorical_cross_entropy_with_scores{I<:Integer}(output::Variable, target::Vector{I}; avg::Bool=true, eps::AbstractFloat=1e-20)
    size(output, 2) == length(target) || throw(DimensionMismatch("output must be size Mx$(length(target)) for Integer target vector with length $(length(target)) but got $(size(output))"))
    pr = softmax(output.data)
    nll = 0.0
    for j = 1:length(target)
        nll -= log(pr[target[j],j] + eps)
    end
    isfinite(nll) || throw(ErrorException("nll not finite ($nll)"))
    return avg ? nll / length(target) : nll
end

function categorical_cross_entropy_with_scores{I<:Integer}(stack::CallbackStack, output::GradVariable, target::Vector{I}; avg::Bool=true, eps::AbstractFloat=1e-20)
    size(output, 2) == length(target) || throw(DimensionMismatch("output must be size Mx$(length(target)) for Integer target vector with length $(length(target)) but got $(size(output))"))
    pr = softmax(output.data)
    nll = 0.0
    weight = avg ? 1.0 / length(target) : 1.0
    for j = 1:size(pr, 2)
        for i = 1:size(pr, 1)
            output.grad[i,j] += weight * (pr[i,j] - (i == target[j] ? 1 : 0))
        end
        nll -= log(pr[target[j], j] + eps)
    end
    isfinite(nll) || throw(ErrorException("nll not finite ($nll)"))
    return avg ? nll / length(target) : nll
end
