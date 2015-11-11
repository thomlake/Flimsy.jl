using Flimsy
import Flimsy: Var
using Base.Test

x = Var(randn(3))
y = tanh(x)
@test size(y) == (3, 1)
@test all(tanh(x.data) .== y.data)
test_op_grad((s)->tanh(s, x), ()->tanh(x), x)

x = Var(randn(3, 5))
y = tanh(x)
@test size(y) == (3, 5)
@test all(tanh(x.data) .== y.data)
test_op_grad((s)->tanh(s, x), ()->tanh(x), x)
