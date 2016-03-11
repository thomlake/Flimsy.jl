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

        params = setup(ValueComponent(value=randn(m, n)); dynamic=true)
        @component cost(params::ValueComponent, target) = Cost.mse(params.value, target)
        @fact check_gradients(cost, params, target; verbose=false) --> true
    end
end
