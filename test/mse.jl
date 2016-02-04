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
        params = ValueComponent(x)
        scope = Scope(params)
        @component cost(p) = Cost.mse(p.value, target)
        c = () -> cost(scope, params)
        g = () -> gradient!(cost, scope, params)
        @fact check_gradients(g, c, params, verbose=false) --> true
    end
end
