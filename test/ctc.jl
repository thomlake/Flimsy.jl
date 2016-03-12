using Flimsy
import Flimsy: Ctc
import Flimsy.Components: VectorComponent

facts("ctc_with_scores") do
    BLANK = 1
    SYMBOLS = [2, 3, 4]
    LANGUAGE = vcat(BLANK, SYMBOLS)

    context("expand") do
        @fact Ctc.expand([2], BLANK)       --> [1, 2, 1]
        @fact Ctc.expand([2, 2, 3], BLANK) --> [1, 2, 1, 2, 1, 3, 1]
    end
    
    context("trim") do
        @fact Ctc.trim([1, 1], BLANK)                   --> []
        @fact Ctc.trim([2], BLANK)                      --> [2]
        @fact Ctc.trim([1, 1, 2, 1], BLANK)             --> [2]
        @fact Ctc.trim([3, 4, 1, 2, 4, 1, 1, 3], BLANK) --> [3, 4, 2, 4, 3]
    end
    
    # triminv
    context("trim_inv") do
        for xs in Ctc.triminv([2, 3], 5, LANGUAGE, BLANK)
            @fact Ctc.trim(xs, BLANK) --> [2, 3]
        end
        for xs in Ctc.triminv([2, 3], 6, LANGUAGE, BLANK)
            @fact Ctc.trim(xs, BLANK) --> [2, 3]
        end
    end


    context("forward") do
        S = 3
        T = 2 * S + 4
        xs = rand(SYMBOLS, S)
        ys = Ctc.expand(xs, BLANK)
        @fact length(ys) --> less_than_or_equal(T)

        scale = 10
        output = [DataVariable(scale .* randn(length(LANGUAGE), 1)) for t = 1:T]
        lpmat = Ctc.make_lpmat(output)

        ll_bf = Ctc.bruteforce(xs, lpmat, LANGUAGE, BLANK)
        nll_ctc = Cost.ctc_with_scores(DynamicScope(), output, xs, BLANK)
        @fact -nll_ctc --> roughly(ll_bf)

        fmat = Ctc.forward(ys, lpmat, BLANK)
        bmat = Ctc.backward(ys, lpmat, BLANK)
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
        targets = rand(SYMBOLS, S)
        params = setup(VectorComponent(values=[randn(LL, 1) for t = 1:T]))
        @component cost(params::VectorComponent, target) = Cost.ctc_with_scores(params.values, targets, BLANK)
        @fact check_gradients(cost, params, targets; verbose=false) --> true
    end
end
