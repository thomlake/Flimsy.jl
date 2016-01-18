"""Mean squared error function. MSE is proportional to the negative log likelihood of a Gaussian random variable.

mse(output::Variable, target::Union{Real,AbstractVector,AbstractMatrix})
"""
function mse end

# 1x1
function mse(stack::CallbackStack, output::GradVariable, target::Real)
    is_scalar(output) || error("mse: output must be 1x1 for scalar target, got $(size(output))")
    delta = output.data[1] - target
    output.grad[1] += delta
    sse = delta * delta
    return 0.5 * sse
end

function mse(output::Variable, target::Real)
    is_scalar(output) || error("mse: output must be 1x1 for scalar target, got $(size(output))")
    delta = output.data[1] - target
    sse = delta * delta
    return 0.5 * sse
end

# Mx1
function mse(stack::CallbackStack, output::Variable, target::AbstractVector)
    is_column_vector(output) || error("mse: output must be Mx1 for vector target, got $(size(output))")
    sse = 0
    for i in eachindex(target)
        delta = output.data[i] - target[i]
        output.grad[i] += delta
        sse += delta * delta
    end
    return 0.5 * sse
end

function mse(output::Variable, target::AbstractVector)
    is_column_vector(output) || error("mse: output must be Mx1 for vector target, got $(size(output))")
    sse = 0
    for i in eachindex(target)
        delta = output.data[i] - target[i]
        sse += delta * delta
    end
    return 0.5 * sse
end

# MxN
function mse(stack::CallbackStack, output::Variable, target::AbstractMatrix)
    size(output) == size(target) || error("mse: output and target sizes must match, got $(size(output)) and $(size(target))")
    sse = 0
    for i in eachindex(target)
        delta = output.data[i] - target[i]
        output.grad[i] += delta
        sse += delta * delta
    end
    return 0.5 * sse
end

function mse(output::Variable, target::AbstractMatrix)
    size(output) == size(target) || error("mse: output and target sizes must match, got $(size(output)) and $(size(target))")
    sse = 0
    for i in eachindex(target)
        delta = output.data[i] - target[i]
        sse += delta * delta
    end
    return 0.5 * sse
end

# # Bernoulli negative log likelihood loss function
# function bern{T,M,N}(target::AbstractArray{Bool}, output::Variable{T,M,N}, eps::AbstractFloat=1e-20)
#     @assert size(target) == (M, N)

#     nll = 0.0
#     for i in eachindex(target)
#         pr_target = target[i] ? output.data[i] : 1 - output.data[i]
#         pr_target = max(eps, pr_target)
#         nll -= log(pr_target)
#     end
#     isfinite(nll) || error("nll: $nll not finite")
#     return nll
# end

# function bern{T,M,N}(stack::BPStack, target::AbstractArray{Bool}, output::Variable{T,M,N}, eps::AbstractFloat=1e-20)
#     @assert size(target) == (M, N)
#     nll = 0.0
#     for i in eachindex(target)
#         pr_target = if target[i]
#             p = output.data[i]
#             p = max(eps, p)
#             output.grad[i] -= 1 / p
#             p
#         else
#             p = 1 - output.data[i]
#             p = max(eps, p)
#             output.grad[i] += 1 / p
#             p
#         end
#         nll -= log(pr_target)
#     end
#     isfinite(nll) || error("nll: $nll not finite")
#     return nll
# end

# bern{T}(target::Bool, output::Variable{T,1,1}, eps=AbstractFloat=1e-20) = bern(fill(target, 1, 1), output, eps)
# bern{T}(stack::BPStack, target::Bool, output::Variable{T,1,1}, eps=AbstractFloat=1e-20) = bern(stack, fill(target, 1, 1), output, eps)

# bern{T}(target::AbstractVector{Bool}, output::Variable{T,1,1}, eps=AbstractFloat=1e-20) = bern(reshape(target, 1, 1), output, eps)
# bern{T}(stack::BPStack, target::AbstractVector{Bool}, output::Variable{T,1,1}, eps=AbstractFloat=1e-20) = bern(stack, reshape(target, 1, 1), output, eps)

# bern{T,M}(target::AbstractVector{Bool}, output::Variable{T,M,1}, eps=AbstractFloat=1e-20) = bern(reshape(target, M, 1), output, eps)
# bern{T,M}(stack::BPStack, target::AbstractVector{Bool}, output::Variable{T,M,1}, eps=AbstractFloat=1e-20) = bern(stack, reshape(target, M, 1), output, eps)

# bern{T,N}(target::AbstractVector{Bool}, output::Variable{T,1,N}, eps=AbstractFloat=1e-20) = bern(reshape(target, 1, N), output, eps)
# bern{T,N}(stack::BPStack, target::AbstractVector{Bool}, output::Variable{T,1,N}, eps=AbstractFloat=1e-20) = bern(stack, reshape(target, 1, N), output, eps)


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
