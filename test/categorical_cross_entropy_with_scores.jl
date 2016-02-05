using Flimsy
import Flimsy.Components: ValueComponent

facts("categorical_cross_entropy_with_scores") do
    K = 4
    context("$(K)x1") do
        for k = 1:K
            params = ValueComponent(value=randn(K, 1))
            scope = DynamicScope(params)
            @component cost() = Cost.categorical_cross_entropy_with_scores(params.value, k)
            g = () -> gradient!(cost, scope)
            c = () -> cost(scope)
            @fact check_gradients(g, c, params, verbose=false) --> true
        end
    end

    n = 7
    context("$(K)x$(n)") do
        targets = rand(1:K, n)
        params = ValueComponent(value=randn(K, n))
        scope = DynamicScope(params)
        @component cost() = Cost.categorical_cross_entropy_with_scores(params.value, targets)
        g = () -> gradient!(cost, scope)
        c = () -> cost(scope)
        @fact check_gradients(g, c, params, verbose=false) --> true
    end
end
