using Flimsy
using Base.Test

m, n = 4, 3
w = GradVariable(randn(m, n))
x = GradVariable(randn(n))
b = GradVariable(randn(m))

y = affine(w, x, b)
@test size(y) == (m, 1)
@test all((w.data * x.data + b.data) .== y.data)
test_op_grad_mse(affine, w, x, b, wrt=[w, x, b])

m, n, k = 5, 10, 7
w = GradVariable(randn(m, n))
x = GradVariable(randn(n, k))
b = GradVariable(randn(m))
y = affine(w, x, b)
@test size(y) == (m, k)
@test all((w.data * x.data .+ b.data) .== y.data)
test_op_grad_mse(affine, w, x, b, wrt=[w, x, b])

w = GradVariable(randn(m, n))
x = GradVariable(randn(n, k))
b = GradVariable(randn(m + 1))
@test_throws OperationError affine(w, x, b)

w = GradVariable(randn(m, n + 1))
x = GradVariable(randn(n, k))
b = GradVariable(randn(m))
@test_throws DimensionMismatch affine(w, x, b)