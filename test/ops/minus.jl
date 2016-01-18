using Flimsy
using Base.Test

# minus(Mx1, Mx1)
m, n = 3, 1
a = GradVariable(randn(m))
b = GradVariable(randn(m))
c = minus(a, b)
@test size(c) == (m, n)
@test all((a.data .- b.data) .== c.data)
test_op_grad_mse(minus, a, b, wrt=[a, b])

# minus(MxN, MxN)
m, n = 3, 5
a = GradVariable(randn(m, n))
b = GradVariable(randn(m, n))
c = minus(a, b)
@test size(c) == (m, n)
@test all((a.data .- b.data) .== c.data)
test_op_grad_mse(minus, a, b, wrt=[a, b])

# minus(MxN, Mx1)
m, n = 3, 5
a = GradVariable(randn(m, n))
b = GradVariable(randn(m))
c = minus(a, b)
@test size(c) == (m, n)
@test all((a.data .- b.data) .== c.data)
test_op_grad_mse(minus, a, b, wrt=[a, b])

# minus(Mx1, MxN)
m, n = 3, 5
a = GradVariable(randn(m))
b = GradVariable(randn(m, n))
c = minus(a, b)
@test size(c) == (m, n)
@test all((a.data .- b.data) .== c.data)
test_op_grad_mse(minus, a, b, wrt=[a, b])
