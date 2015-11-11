using Flimsy
import Flimsy: Var
using Base.Test

x = Var(randn(3))
y = wta(x)
@test size(y) == (3, 1)
xmax, imax = findmax(x.data)
for i = 1:3
    @test y.data[i] == (i == imax ?  xmax : 0)
end
test_op_grad((s)->wta(s, x), ()->wta(x), x)

x = Var(randn(3, 5))
y = wta(x)
@test size(y) == (3, 5)
for j = 1:5
    xmax, imax = findmax(x.data[:,j])
    for i = 1:3
        @test y.data[i,j] == (i == imax ?  xmax : 0)
    end
end
test_op_grad((s)->wta(s, x), ()->wta(x), x)
