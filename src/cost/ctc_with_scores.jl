import ..Flimsy: Ctc

# Connectionist Temporal Classification loss
function ctc_with_scores{V<:Variable, I<:Integer}(scope::Scope, output::Vector{V}, target::Vector{I}, blank::Int)
    ys = Ctc.expand(target, blank)
    T = length(output)
    S = size(output[1], 1)

    lpmat = Ctc.make_lpmat(output)
    fmat = Ctc.forward(ys, lpmat, blank)
    ll = Flimsy.Extras.logsumexp(fmat[end, end], fmat[end-1, end])
    return -ll
end

function ctc_with_scores{V<:GradVariable,I<:Integer}(scope::GradScope, output::Vector{V}, target::Vector{I}, blank::Int)
    ys = Ctc.expand(target, blank)
    T = length(output)
    S = size(output[1], 1)

    length(ys) <= T || throw(Ctc.CtcError("INPUT ERROR", "number of expanded outputs > number of timesteps"))

    lpmat = Ctc.make_lpmat(output)
    fmat = Ctc.forward(ys, lpmat, blank)
    bmat = Ctc.backward(ys, lpmat, blank)
    fbmat = fmat + bmat
    ll = Flimsy.Extras.logsumexp(fmat[end, end], fmat[end-1, end])

    if !isfinite(ll)
        msg = [
            "ll not finite ($ll, $(length(target)), $(length(output)))",
            "forward entries: (fmat[end, end],fmat[end-1, end]) = $((fmat[end,end], fmat[end-1, end]))",
        ]
        throw(Ctc.CtcError("NOT FINITE", join(msg, "\n")))
    end

    for t = 1:T
        for k = 1:S
            total = -Inf
            for s in findin(ys, k)
                total = Flimsy.Extras.logsumexp(total, fbmat[s, t])
            end
            g = exp(lpmat[k,t]) - exp(total - ll)
            isfinite(g) || throw(Ctc.CtcError("NOT FINITE", "gradient not finite: $g"))
            output[t].grad[k] += g 
        end
    end
    return -ll
end