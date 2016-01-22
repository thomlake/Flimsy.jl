using Flimsy
using Base.Test

function test_softmax()
    # Mx1
    m, n = 5, 1
    x = randn(m)
    y = softmax(x)
    @test isa(y, Vector)
    @test size(y) == (m,)
    @test_approx_eq exp(x) / sum(exp(x)) y
    @test_approx_eq sum(y) 1

    x = DataVariable(randn(m))
    y = softmax(x)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test_approx_eq softmax(x.data) y.data
    @test_approx_eq sum(y.data) 1

    x = GradVariable(randn(m))
    y = softmax(x)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test_approx_eq softmax(x.data) y.data
    @test_approx_eq sum(y.data) 1

    x = DataVariable(randn(m))
    y = softmax(CallbackStack(), x)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test_approx_eq softmax(x.data) y.data
    @test_approx_eq sum(y.data) 1

    x = GradVariable(randn(m))
    y = softmax(CallbackStack(), x)
    @test isa(y, GradVariable)
    @test size(y) == (m, n)
    @test_approx_eq softmax(x.data) y.data
    @test_approx_eq sum(y.data) 1

    x = GradVariable(randn(m))
    test_op_grad_mse(softmax, x, wrt=x)

    # MxN
    m, n = 4, 7
    x = randn(m, n)
    y = softmax(x)
    @test isa(y, Matrix)
    @test size(y) == (m, n)
    @test_approx_eq exp(x) ./ sum(exp(x), 1) y
    @test_approx_eq sum(y, 1) ones(n)
    @test_approx_eq sum(y) n

    x = DataVariable(randn(m, n))
    y = softmax(x)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test_approx_eq softmax(x.data) y.data
    @test_approx_eq sum(y.data, 1) ones(n)
    @test_approx_eq sum(y.data) n

    x = GradVariable(randn(m, n))
    y = softmax(x)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test_approx_eq softmax(x.data) y.data
    @test_approx_eq sum(y.data, 1) ones(n)
    @test_approx_eq sum(y.data) n

    x = DataVariable(randn(m, n))
    y = softmax(CallbackStack(), x)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test_approx_eq softmax(x.data) y.data
    @test_approx_eq sum(y.data, 1) ones(n)
    @test_approx_eq sum(y.data) n

    x = GradVariable(randn(m, n))
    y = softmax(CallbackStack(), x)
    @test isa(y, GradVariable)
    @test size(y) == (m, n)
    @test_approx_eq softmax(x.data) y.data
    @test_approx_eq sum(y.data, 1) ones(n)
    @test_approx_eq sum(y.data) n

    x = GradVariable(randn(m, n))
    test_op_grad_mse(softmax, x, wrt=x)
end
test_softmax()
