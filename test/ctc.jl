using Flimsy
import Flimsy: CTC

facts("ctc_with_scores") do
    BLANK = 1
    SYMBOLS = [2, 3, 4]
    LANGUAGE = vcat(BLANK, SYMBOLS)

    context("expand") do
        @fact CTC.expand([2], BLANK)       --> [1, 2, 1]
        @fact CTC.expand([2, 2, 3], BLANK) --> [1, 2, 1, 2, 1, 3, 1]
    end
    
    context("trim") do
        @fact CTC.trim([1, 1], BLANK)                   --> []
        @fact CTC.trim([2], BLANK)                      --> [2]
        @fact CTC.trim([1, 1, 2, 1], BLANK)             --> [2]
        @fact CTC.trim([3, 4, 1, 2, 4, 1, 1, 3], BLANK) --> [3, 4, 2, 4, 3]
    end
    
    # triminv
    context("trim_inv") do
        for xs in CTC.triminv([2, 3], 5, LANGUAGE, BLANK)
            @fact CTC.trim(xs, BLANK) --> [2, 3]
        end
        for xs in CTC.triminv([2, 3], 6, LANGUAGE, BLANK)
            @fact CTC.trim(xs, BLANK) --> [2, 3]
        end
    end


    context("forward") do
        S = 3
        T = 2 * S + 4
        xs = rand(SYMBOLS, S)
        ys = CTC.expand(xs, BLANK)
        @fact length(ys) --> less_than_or_equal(T)

        scale = 10
        output = [DataVariable(scale .* randn(length(LANGUAGE), 1)) for t = 1:T]
        lpmat = CTC.make_lpmat(output)

        ll_bf = CTC.bruteforce(xs, lpmat, LANGUAGE, BLANK)
        nll_ctc = Cost.ctc_with_scores(DynamicScope(), output, xs, BLANK)
        @fact -nll_ctc --> roughly(ll_bf)

        fmat = CTC.forward(ys, lpmat, BLANK)
        bmat = CTC.backward(ys, lpmat, BLANK)
        fbmat = fmat + bmat
        for t = 1:T
            @fact Flimsy.Extras.logsumexp(fbmat[:,t]) --> roughly(ll_bf)
        end
    end

    context("gradient") do
        SYMBOLS = collect(2:20)
        LANGUAGE = vcat(BLANK, SYMBOLS)
        LL = length(LANGUAGE)
        S = 5
        T = 20
        xs = rand(SYMBOLS, S)
        output = [GradVariable(randn(LL, 1), zeros(LL, 1)) for t = 1:T]
        scope = DynamicScope()

        const eps = 1e-6
        const tol = 1e-6
        gradient!(Cost.ctc_with_scores, scope, output, xs, BLANK)
        for t = 1:T
            param = output[t]
            for i = 1:size(param, 1)
                xi = param.data[i]
                param.data[i] = xi + eps
                lp = Cost.ctc_with_scores(scope, output, xs, BLANK)
                param.data[i] = xi - eps
                lm = Cost.ctc_with_scores(scope, output, xs, BLANK)
                param.data[i] = xi
                dxi = (lp - lm) / (2 * eps)
                @fact dxi - param.grad[i] --> roughly(0, tol)
            end
        end
    end
end
