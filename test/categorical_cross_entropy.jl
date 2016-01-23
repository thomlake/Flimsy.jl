using Flimsy
import Flimsy.Components: ValueComponent
using Base.Test

function test_categorical_cross_entropy()
    K = 4
    for k = 1:K
        x = GradVariable(randn(K))
        p = ValueComponent(x)
        @component cost() = Cost.categorical_cross_entropy(softmax(p.value), k)
        grad = () -> gradient!(cost)
        check_gradients(grad, cost, p, verbose=false)
    end

    n = 7
    targets = rand(1:K, n)
    x = GradVariable(randn(K, n))
    p = ValueComponent(x)
    @component cost() = Cost.categorical_cross_entropy(softmax(p.value), targets)
    grad = () -> gradient!(cost)
    check_gradients(grad, cost, p, verbose=false)
end
test_categorical_cross_entropy()
