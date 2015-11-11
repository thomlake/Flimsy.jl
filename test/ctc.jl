using Flimsy
using Base.Test

const BLANK = 1
const SYMBOLS = [2, 3, 4]
const LANGUAGE = vcat(BLANK, SYMBOLS)

# expand
@test Flimsy.Cost.CTC.expand([2], BLANK) == [1, 2, 1]
@test Flimsy.Cost.CTC.expand([2, 2, 3], BLANK) == [1, 2, 1, 2, 1, 3, 1]

# trim
@assert Flimsy.Cost.CTC.trim([1, 1], BLANK) == []
@assert Flimsy.Cost.CTC.trim([2], BLANK) == [2]
@assert Flimsy.Cost.CTC.trim([1, 1, 2, 1], BLANK) == [2]
@assert Flimsy.Cost.CTC.trim([3, 4, 1, 2, 4, 1, 1, 3], BLANK) == [3, 4, 2, 4, 3]

# triminv
for xs in Flimsy.Cost.CTC.triminv([2, 3], 5, LANGUAGE, BLANK)
    @assert Flimsy.Cost.CTC.trim(xs, BLANK) == [2, 3]
end
for xs in Flimsy.Cost.CTC.triminv([2, 3], 6, LANGUAGE, BLANK)
    @assert Flimsy.Cost.CTC.trim(xs, BLANK) == [2, 3]
end


# forward
S = 3
T = 2 * S + 4
xs = rand(SYMBOLS, S)
ys = Flimsy.Cost.CTC.expand(xs, BLANK)
@assert T >= length(ys)

outputs = [Var(randn(length(LANGUAGE), 1)) for t = 1:T]
lpmat = Flimsy.Cost.CTC.make_lpmat(outputs)
ll_bf = Flimsy.Cost.CTC.bruteforce(xs, lpmat, LANGUAGE, BLANK)
nll_ctc = Flimsy.Cost.ctc(xs, outputs, BLANK)
@test_approx_eq ll_bf -nll_ctc

fmat = Flimsy.Cost.CTC.forward(ys, lpmat, BLANK)
bmat = Flimsy.Cost.CTC.backward(ys, lpmat, BLANK)
fbmat = fmat + bmat
for t = 1:T
    @test_approx_eq ll_bf Flimsy.Extras.logsumexp(fbmat[:,t])
end

# gradients
const eps = 1e-6
const tol = 1e-6
Flimsy.Cost.ctc(BPStack(), xs, outputs, BLANK)
for t = 1:T
    param = outputs[t]
    for i = 1:size(param, 1)
        xi = param.data[i]
        param.data[i] = xi + eps
        lp = Flimsy.Cost.ctc(xs, outputs, BLANK)
        param.data[i] = xi - eps
        lm = Flimsy.Cost.ctc(xs, outputs, BLANK)
        param.data[i] = xi
        dxi = (lp - lm) / (2 * eps)
        if abs(dxi - param.grad[i]) > tol
            errmsg = "Finite difference gradient check failed!"
            errelm = "  time => $t, index => $i, ratio => $(dxi / param.grad[i])"
            errdsc = "  |$(dxi) - $(param.grad[i])| > $tol"
            println("$errmsg\n$errelm\n$errdsc")
        end
    end
end
