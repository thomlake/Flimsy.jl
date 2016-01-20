using Flimsy
import Flimsy.Components: ValueComponent
using Base.Test

K = 4
for k = 1:K
    x = GradVariable(randn(K))
    p = ValueComponent(x)
    @component f() = Cost.categorical_cross_entropy(softmax(p.value), k)
    g = () -> gradient!(f)
    c = () -> f()
    check_gradients(g, c, p, verbose=false)
end
