using Flimsy
import Flimsy.Components: ValueComponent

facts("categorical_cross_entropy_with_scores") do
    K = 4
    context("$(K)x1") do
        for k = 1:K
            params = Model(ValueComponent(value=randn(K, 1)), DynamicScope())
            @component cost(params::ValueComponent, target) = Cost.categorical_cross_entropy_with_scores(params.value, target)
            @fact check_gradients(cost, params, k; verbose=false) --> true
        end
    end

    n = 7
    context("$(K)x$(n)") do
        targets = rand(1:K, n)
        params = Model(ValueComponent(value=randn(K, n)), DynamicScope())
        @component cost(params::ValueComponent, target) = Cost.categorical_cross_entropy_with_scores(params.value, target)
        @fact check_gradients(cost, params, targets; verbose=false) --> true
    end
end
