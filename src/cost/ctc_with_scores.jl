import ..Flimsy: CTC

# Connectionist Temporal Classification loss
function ctc_with_scores{V<:Variable, I<:Int}(output::Vector{V}, target::Vector{I}, blank::Int)
    ys = CTC.expand(target, blank)
    T = length(output)
    S = size(output[1], 1)

    lpmat = CTC.make_lpmat(output)
    fmat = CTC.forward(ys, lpmat, blank)
    ll = Flimsy.Extras.logsumexp(fmat[end, end], fmat[end-1, end])
    return -ll
end

function ctc_with_scores{V<:GradVariable,I<:Integer}(stack::CallbackStack, output::Vector{V}, target::Vector{I}, blank::Int)
    ys = CTC.expand(target, blank)
    T = length(output)
    S = size(output[1], 1)

    length(ys) <= T || throw(CTC.CTCError("INPUT ERROR", "number of expanded outputs > number of timesteps"))

    lpmat = CTC.make_lpmat(output)
    fmat = CTC.forward(ys, lpmat, blank)
    bmat = CTC.backward(ys, lpmat, blank)
    fbmat = fmat + bmat
    ll = Flimsy.Extras.logsumexp(fmat[end, end], fmat[end-1, end])

    if !isfinite(ll)
        msg = [
            "ll not finite ($ll, $(length(target)), $(length(output)))",
            "forward entries: (fmat[end, end],fmat[end-1, end]) = $((fmat[end,end], fmat[end-1, end]))",
        ]
        throw(CTC.CTCError("NOT FINITE", join(msg, "\n")))
    end

    for t = 1:T
        for k = 1:S
            total = -Inf
            for s in findin(ys, k)
                total = Flimsy.Extras.logsumexp(total, fbmat[s, t])
            end
            g = exp(lpmat[k,t]) - exp(total - ll)
            isfinite(g) || throw(CTC.CTCError("NOT FINITE", "gradient not finite: $g"))
            output[t].grad[k] += g 
        end
    end
    return -ll
end