using Flimsy
import Flimsy.Components: ValueComponent
using Base.Test

function test_mse()
    for (m, n) in [(1,1), (3, 1), (4, 5)]
        x = GradVariable(randn(m, n))
        target = if (m, n) == (1, 1)
            randn()
        elseif n == 1
            randn(n)
        else
            randn(m, n)
        end
        p = ValueComponent(x)
        @component cost() = Cost.mse(p.value, target)
        grad = () -> gradient!(cost)
        check_gradients(grad, cost, p, verbose=false)
    end
end
test_mse()
