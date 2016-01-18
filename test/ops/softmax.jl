using Flimsy
using Base.Test

# Column Vector
x = randn(5)
y = softmax(x)
@test size(y) == (5,)
@test isapprox(sum(y), 1)

x = GradVariable(x)
y = softmax(x)
@test size(y) == (5, 1)
@test all(softmax(x.data) .== y.data)
@test isapprox(sum(y.data), 1)
test_op_grad_mse(softmax, x, wrt=x)

# Nx1 Matrix
x = randn(5, 1)
y = softmax(x)
@test size(y) == (5, 1)
@test isapprox(sum(y), 1)

x = GradVariable(x)
y = softmax(x)
@test size(y) == (5, 1)
@test all(softmax(x.data) .== y.data)
@test isapprox(sum(y.data), 1)
test_op_grad_mse(softmax, x, wrt=x)

# NxM Matrix
x = randn(4, 5)
y = softmax(x)
@test size(y) == (4, 5)
for y_j in sum(y, 1)
    @test isapprox(y_j, 1)
end

x = GradVariable(x)
y = softmax(x)
@test size(y) == (4, 5)
@test all(softmax(x.data) .== y.data)
for y_j in sum(y.data, 1)
    @test isapprox(y_j, 1)
end
test_op_grad_mse(softmax, x, wrt=x)
