using Flimsy
using Base.Test

function test_sigmoid()
    # Mx1
    m, n = 3, 1
    x = DataVariable(randn(m))
    y = sigmoid(x)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test all(sigmoid(x.data) .== y.data)

    x = GradVariable(randn(m))
    y = sigmoid(x)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test all(sigmoid(x.data) .== y.data)

    x = DataVariable(randn(m))
    y = sigmoid(CallbackStack(), x)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test all(sigmoid(x.data) .== y.data)

    x = GradVariable(randn(m))
    y = sigmoid(CallbackStack(), x)
    @test isa(y, GradVariable)
    @test size(y) == (m, n)
    @test all(sigmoid(x.data) .== y.data)

    x = GradVariable(randn(m))
    test_op_grad_mse(sigmoid, x, wrt=x)

    # MxN
    m, n = 5, 8
    x = DataVariable(randn(m, n))
    y = sigmoid(x)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test all(sigmoid(x.data) .== y.data)

    x = GradVariable(randn(m, n))
    y = sigmoid(x)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test all(sigmoid(x.data) .== y.data)

    x = DataVariable(randn(m, n))
    y = sigmoid(CallbackStack(), x)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test all(sigmoid(x.data) .== y.data)

    x = GradVariable(randn(m, n))
    y = sigmoid(CallbackStack(), x)
    @test isa(y, GradVariable)
    @test size(y) == (m, n)
    @test all(sigmoid(x.data) .== y.data)

    x = GradVariable(randn(m, n))
    test_op_grad_mse(sigmoid, x, wrt=x)
end
test_sigmoid()
