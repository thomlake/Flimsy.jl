using Flimsy
import Flimsy: Var
using Base.Test

x = Var(randn(3))
y = relu(x)
@test size(y) == (3, 1)
@test all(max(0, x.data) .== y.data)
test_op_grad((s)->relu(s, x), ()->relu(x), x)

x = Var(randn(3, 5))
y = relu(x)
@test size(y) == (3, 5)
@test all(max(0, x.data) .== y.data)
test_op_grad((s)->relu(s, x), ()->relu(x), x)
