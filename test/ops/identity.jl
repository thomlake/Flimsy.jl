using Flimsy
import Flimsy: Var
using Base.Test

x = Var(randn(3))
y = identity(x)
@test size(y) == (3, 1)
@test all(x.data .== y.data)
test_op_grad((s)->identity(s, x), ()->identity(x), x)

x = Var(randn(3, 5))
y = identity(x)
@test size(y) == (3, 5)
@test all(x.data .== y.data)
test_op_grad((s)->identity(s, x), ()->identity(x), x)
