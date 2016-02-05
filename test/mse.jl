using Flimsy
import Flimsy.Components: ValueComponent

facts("mse") do
    for (m, n) in [(1,1), (3, 1), (4, 5)]
        target = if (m, n) == (1, 1)
            randn()
        elseif n == 1
            randn(n)
        else
            randn(m, n)
        end

        params = ValueComponent(value=randn(m, n))
        scope = DynamicScope()
        @component cost() = Cost.mse(params.value, target)
        g = () -> gradient!(cost, scope)
        c = () -> cost(scope)
        @fact check_gradients(g, c, params, verbose=false) --> true
    end
end
