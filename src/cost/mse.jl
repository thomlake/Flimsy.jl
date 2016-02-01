"""
**Mean squared Error**

MSE is proportional to the negative log likelihood of a Gaussian random variable.

mse(output::Variable, target::Union{Real,AbstractVector,AbstractMatrix})
mse(output::Variable, target::Union{Real,AbstractVector,AbstractMatrix}, weight::Real)
"""
function mse end

# -------- #
# 1x1 Real #
# -------- #
function mse(scope::Scope, output::Variable, target::Real)
    is_scalar(output) || throw(DimensionMismatch("mse: output must be 1x1 for scalar target, got $(size(output))"))
    delta = output.data[1] - target
    sse = delta * delta
    return 0.5 * sse
end

function mse(scope::GradScope, output::GradVariable, target::Real)
    is_scalar(output) || throw(DimensionMismatch("mse: output must be 1x1 for scalar target, got $(size(output))"))
    delta = output.data[1] - target
    output.grad[1] += delta
    sse = delta * delta
    return 0.5 * sse
end

mse(scope::Scope, output::Variable, target::Real, weight::Real) = weight * mse(output, target)

function mse(scope::GradScope, output::GradVariable, target::Real, weight::Real)
    is_scalar(output) || throw(DimensionMismatch("mse: output must be 1x1 for scalar target, got $(size(output))"))
    delta = output.data[1] - target
    output.grad[1] += weight * delta
    sse = delta * delta
    return 0.5 * weight * sse
end


# ---------------- #
# Mx1 Vector{Real} #
# ---------------- #
function mse(scope::Scope, output::Variable, target::AbstractVector)
    is_column_vector(output) || throw(DimensionMismatch("mse: output must be Mx1 for vector target, got $(size(output))"))
    sse = 0
    for i in eachindex(target)
        delta = output.data[i] - target[i]
        sse += delta * delta
    end
    return 0.5 * sse
end

function mse(scope::GradScope, output::GradVariable, target::AbstractVector)
    is_column_vector(output) || throw(DimensionMismatch("mse: output must be Mx1 for vector target, got $(size(output))"))
    sse = 0
    for i in eachindex(target)
        delta = output.data[i] - target[i]
        output.grad[i] += delta
        sse += delta * delta
    end
    return 0.5 * sse
end

mse(scope::Scope, output::Variable, target::AbstractVector, weight::Real) = weight * mse(output, target)

function mse(scope::GradScope, output::GradVariable, target::AbstractVector, weight::Real)
    is_column_vector(output) || throw(DimensionMismatch("mse: output must be Mx1 for vector target, got $(size(output))"))
    sse = 0
    for i in eachindex(target)
        delta = output.data[i] - target[i]
        output.grad[i] += weight * delta
        sse += delta * delta
    end
    return 0.5 * weight * sse
end


# ---------------- #
# Mx1 Matrix{Real} #
# ---------------- #
function mse(scope::Scope, output::Variable, target::AbstractMatrix)
    size(output) == size(target) || throw(DimensionMismatch("mse: output and target sizes must match, got $(size(output)) and $(size(target))"))
    sse = 0
    for i in eachindex(target)
        delta = output.data[i] - target[i]
        sse += delta * delta
    end
    return 0.5 * sse
end

function mse(scope::GradScope, output::GradVariable, target::AbstractMatrix)
    size(output) == size(target) || throw(DimensionMismatch("mse: output and target sizes must match, got $(size(output)) and $(size(target))"))
    sse = 0
    for i in eachindex(target)
        delta = output.data[i] - target[i]
        output.grad[i] += delta
        sse += delta * delta
    end
    return 0.5 * sse
end

mse(scope::Scope, output::Variable, target::AbstractMatrix, weight::Real) = weight * mse(output, target)

function mse(scope::GradScope, output::Variable, target::AbstractMatrix, weight::Real)
    size(output) == size(target) || throw(DimensionMismatch("mse: output and target sizes must match, got $(size(output)) and $(size(target))"))
    sse = 0
    for i in eachindex(target)
        delta = output.data[i] - target[i]
        output.grad[i] += weight * delta
        sse += delta * delta
    end
    return 0.5 * weight * sse
end


# # Max Margin loss
# function margin{T,M}(target::Integer, output::Variable{T,M,1}, m::Real=1)
#     k = Flimsy.Extras.argmaxneq(output, [target])
#     err = max(0, output.data[k] + m - output.data[target])
#     return err
# end

# function margin{T,M}(stack::BPStack, target::Integer, output::Variable{T,M,1}, m::Real=1)
#     k = Flimsy.Extras.argmaxneq(output, [target])
#     err = max(0, output.data[k] + m - output.data[target])
#     if err > 0
#         output.grad[k] += 1
#         output.grad[target] -= 1
#     end 
#     return err
# end

# function margin{I<:Integer}(target::Vector{I}, output::Variable, m::Real=1)
#     @assert size(output, 2) == length(target)
    
#     errsum = 0.0
#     ks = Flimsy.Extras.argmaxneq(output, target)
#     for i = 1:size(output, 2)
#         errsum += max(0, output.data[ks[i], i] + m - output.data[target[i], i])
#     end
#     return errsum
# end

# function margin{I<:Integer}(stack::BPStack, target::Vector{I}, output::Variable, m::Real=1)
#     @assert size(output, 2) == length(target)

#     errsum = 0.0
#     ks = Flimsy.Extras.argmaxneq(output, target)
#     for i = 1:size(output, 2)
#         err = max(0, output.data[ks[i], i] + m - output.data[target[i], i])
#         if err > 0
#             output.grad[ks[i], i] += 1
#             output.grad[target[i], i] -= 1
#             errsum += err
#         end 
#     end
#     return errsum
# end


# # Connectionist Temporal Classification loss
# function ctc{I<:Int,V<:Variable}(target::Vector{I}, output::Vector{V}, blank::Int)
#     ys = CTC.expand(target, blank)
#     T = length(output)
#     S = size(output[1], 1)

#     lpmat = CTC.make_lpmat(output)
#     fmat = CTC.forward(ys, lpmat, blank)
#     ll = Flimsy.Extras.logsumexp(fmat[end, end], fmat[end-1, end])
#     return -ll
# end

# function ctc{I<:Integer,V<:Variable}(stack::BPStack, target::Vector{I}, output::Vector{V}, blank::Int)
#     ys = CTC.expand(target, blank)
#     T = length(output)
#     S = size(output[1], 1)

#     length(ys) <= T || throw(CTC.CTCError("INPUT ERROR", "number of expanded outputs > number of timesteps"))

#     lpmat = CTC.make_lpmat(output)
#     fmat = CTC.forward(ys, lpmat, blank)
#     bmat = CTC.backward(ys, lpmat, blank)
#     fbmat = fmat + bmat
#     ll = Flimsy.Extras.logsumexp(fmat[end, end], fmat[end-1, end])

#     if !isfinite(ll)
#         msg = [
#             "ll not finite ($ll, $(length(target)), $(length(output)))",
#             "forward entries: (fmat[end, end],fmat[end-1, end]) = $((fmat[end,end], fmat[end-1, end]))",
#         ]
#         throw(CTC.CTCError("NOT FINITE", join(msg, "\n")))
#     end

#     for t = 1:T
#         for k = 1:S
#             total = -Inf
#             for s in findin(ys, k)
#                 total = Flimsy.Extras.logsumexp(total, fbmat[s, t])
#             end
#             g = exp(lpmat[k,t]) - exp(total - ll)
#             isfinite(g) || throw(CTC.CTCError("NOT FINITE", "gradient not finite: $g"))
#             output[t].grad[k] += g 
#         end
#     end
#     return -ll
# end

# # REINFORCE (Williams, 1992)
# function reinforce{T,M}(stack::BPStack, action::Integer, probs::Variable{T,M,1}, reward::AbstractFloat, eps::AbstractFloat=1e-20)
#     pr_action = probs.data[action] + eps
#     probs.grad[action] -= reward / pr_action
# end
