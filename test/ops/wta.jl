using Flimsy
using Base.Test

m = 10
x = GradVariable(randn(m))
y = wta(x)
@test size(y) == (m, 1)
xmax, imax = findmax(x.data)
for i = 1:m
    @test y.data[i] == (i == imax ?  xmax : 0)
end
test_op_grad_mse(wta, x, wrt=x)

m, n = 10, 12
x = GradVariable(randn(m, n))
y = wta(x)
@test size(y) == (m, n)
xmax, imax = findmax(x.data)
for j = 1:n
    xmax, imax = findmax(x.data[:,j])
    for i = 1:m
        @test y.data[i,j] == (i == imax ?  xmax : 0)
    end
end
test_op_grad_mse(wta, x, wrt=x)
