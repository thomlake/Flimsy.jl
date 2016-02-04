using Flimsy
import Flimsy.Components: ValueComponent

facts("mse") do
    for (m, n) in [(1,1), (3, 1), (4, 5)]
        x = GradVariable(randn(m, n), zeros(m, n))
        target = if (m, n) == (1, 1)
            randn()
        elseif n == 1
            randn(n)
        else
            randn(m, n)
        end
        p = ValueComponent(x)
        scope = Scope(p)
        @component cost(params) = Cost.mse(params.value, target)
        c = () -> cost(p)
        g = () -> gradient!(cost, scope)
        check_gradients(g, c, scope, verbose=false)
    end
end
