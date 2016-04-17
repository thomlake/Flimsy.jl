using Flimsy
import Flimsy.Components: ValueComponent

facts("categorical_cross_entropy") do
    K = 4
    context("$(K)x1") do
        for k = 1:K
            params = Runtime(ValueComponent(value=randn(K, 1)))
            @comp cost(params::ValueComponent, target) = Cost.categorical_cross_entropy(softmax(params.value), target)
            @fact check_gradients(cost, params, k; verbose=false) --> true
        end
    end

    n = 7
    context("$(K)x$(n)") do
        targets = rand(1:K, n)
        params = Runtime(ValueComponent(value=randn(K, n)))
        @comp cost(params::ValueComponent, target) = Cost.categorical_cross_entropy(softmax(params.value), target)
        @fact check_gradients(cost, params, targets; verbose=false) --> true
    end
end
