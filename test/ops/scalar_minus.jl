using Flimsy
using Base.Test

# minus(a, Mx1)
m, n = 3, 1
a = 0.5
x = GradVariable(randn(m))
y = minus(a, x)
@test size(y) == (m, n)
@test all(a .- x.data .== y.data)
test_op_grad_mse(minus, a, x, wrt=x)

# minus(-a, Mx1)
a = -0.5
x = GradVariable(randn(m))
y = minus(a, x)
@test size(y) == (m, n)
@test all(a .- x.data .== y.data)
test_op_grad_mse(minus, a, x, wrt=x)

# minus(a, MxN)
m, n = 3, 5
a = 0.5
x = GradVariable(randn(m, n))
y = minus(a, x)
@test size(y) == (m, n)
@test all(a .- x.data .== y.data)
test_op_grad_mse(minus, a, x, wrt=x)

# minus(-a, MxN)
a = -0.5
x = GradVariable(randn(m, n))
y = minus(a, x)
@test size(y) == (m, n)
@test all(a .- x.data .== y.data)
test_op_grad_mse(minus, a, x, wrt=x)

# - reverse argument order - #
# minus(a, Mx1)
m, n = 3, 1
a = 0.5
x = GradVariable(randn(m))
y = minus(x, a)
@test size(y) == (m, n)
@test all(x.data .- a .== y.data)
test_op_grad_mse(minus, x, a, wrt=x)

# minus(-a, Mx1)
a = -0.5
x = GradVariable(randn(m))
y = minus(x, a)
@test size(y) == (m, n)
@test all(x.data .- a .== y.data)
test_op_grad_mse(minus, x, a, wrt=x)

# minus(a, MxN)
m, n = 3, 5
a = 0.5
x = GradVariable(randn(m, n))
y = minus(x, a)
@test size(y) == (m, n)
@test all(x.data .- a .== y.data)
test_op_grad_mse(minus, x, a, wrt=x)

# minus(-a, MxN)
a = -0.5
x = GradVariable(randn(m, n))
y = minus(x, a)
@test size(y) == (m, n)
@test all(x.data .- a .== y.data)
test_op_grad_mse(minus, x, a, wrt=x)
