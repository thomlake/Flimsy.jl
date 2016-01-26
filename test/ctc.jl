using Flimsy
import Flimsy: CTC
using Base.Test

function test_ctc()
    BLANK = 1
    SYMBOLS = [2, 3, 4]
    LANGUAGE = vcat(BLANK, SYMBOLS)

    # expand
    @test CTC.expand([2], BLANK) == [1, 2, 1]
    @test CTC.expand([2, 2, 3], BLANK) == [1, 2, 1, 2, 1, 3, 1]

    # trim
    @test CTC.trim([1, 1], BLANK) == []
    @test CTC.trim([2], BLANK) == [2]
    @test CTC.trim([1, 1, 2, 1], BLANK) == [2]
    @test CTC.trim([3, 4, 1, 2, 4, 1, 1, 3], BLANK) == [3, 4, 2, 4, 3]

    # triminv
    for xs in CTC.triminv([2, 3], 5, LANGUAGE, BLANK)
        @test CTC.trim(xs, BLANK) == [2, 3]
    end
    for xs in CTC.triminv([2, 3], 6, LANGUAGE, BLANK)
        @test CTC.trim(xs, BLANK) == [2, 3]
    end


    # forward
    S = 3
    T = 2 * S + 4
    xs = rand(SYMBOLS, S)
    ys = CTC.expand(xs, BLANK)
    @test T >= length(ys)

    scale = 10
    output = [DataVariable(scale .* randn(length(LANGUAGE))) for t = 1:T]
    lpmat = CTC.make_lpmat(output)

    ll_bf = CTC.bruteforce(xs, lpmat, LANGUAGE, BLANK)
    nll_ctc = Cost.ctc_with_scores(output, xs, BLANK)
    @test_approx_eq ll_bf -nll_ctc

    fmat = CTC.forward(ys, lpmat, BLANK)
    bmat = CTC.backward(ys, lpmat, BLANK)
    fbmat = fmat + bmat
    for t = 1:T
        @test_approx_eq ll_bf Flimsy.Extras.logsumexp(fbmat[:,t])
    end

    # gradients
    SYMBOLS = collect(2:20)
    LANGUAGE = vcat(BLANK, SYMBOLS)
    S = 5
    T = 20
    xs = rand(SYMBOLS, S)
    output = [GradVariable(100 .* randn(length(LANGUAGE), 1)) for t = 1:T]
    const eps = 1e-6
    const tol = 1e-6
    Cost.ctc_with_scores(CallbackStack(), output, xs, BLANK)
    for t = 1:T
        param = output[t]
        for i = 1:size(param, 1)
            xi = param.data[i]
            param.data[i] = xi + eps
            lp = Cost.ctc_with_scores(output, xs, BLANK)
            param.data[i] = xi - eps
            lm = Cost.ctc_with_scores(output, xs, BLANK)
            param.data[i] = xi
            dxi = (lp - lm) / (2 * eps)
            if abs(dxi - param.grad[i]) > tol
                errmsg = "Finite difference gradient check failed!"
                errelm = "  time => $t, index => $i, ratio => $(dxi / param.grad[i])"
                errdsc = "  |$(dxi) - $(param.grad[i])| > $tol"
                error("$errmsg\n$errelm\n$errdsc")
            end
        end
    end
end
test_ctc()
