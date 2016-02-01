using Flimsy
using Base.Test

function test_tanh()
    x = GradVariable(randn(10))
    test_op_grad_mse(tanh, x, wrt=x)

    # # Mx1
    # m, n = 3, 1
    # x = DataVariable(randn(m))
    # y = tanh(x)
    # @test isa(y, DataVariable)
    # @test size(y) == (m, n)
    # @test all(tanh(x.data) .== y.data)

    # x = GradVariable(randn(m))
    # y = tanh(x)
    # @test isa(y, DataVariable)
    # @test size(y) == (m, n)
    # @test all(tanh(x.data) .== y.data)

    # x = DataVariable(randn(m))
    # y = tanh(CallbackStack(), x)
    # @test isa(y, DataVariable)
    # @test size(y) == (m, n)
    # @test all(tanh(x.data) .== y.data)

    # x = GradVariable(randn(m))
    # y = tanh(CallbackStack(), x)
    # @test isa(y, GradVariable)
    # @test size(y) == (m, n)
    # @test all(tanh(x.data) .== y.data)

    # x = GradVariable(randn(m))
    # test_op_grad_mse(tanh, x, wrt=x)

    # # MxN
    # m, n = 5, 8
    # x = DataVariable(randn(m, n))
    # y = tanh(x)
    # @test isa(y, DataVariable)
    # @test size(y) == (m, n)
    # @test all(tanh(x.data) .== y.data)

    # x = GradVariable(randn(m, n))
    # y = tanh(x)
    # @test isa(y, DataVariable)
    # @test size(y) == (m, n)
    # @test all(tanh(x.data) .== y.data)

    # x = DataVariable(randn(m, n))
    # y = tanh(CallbackStack(), x)
    # @test isa(y, DataVariable)
    # @test size(y) == (m, n)
    # @test all(tanh(x.data) .== y.data)

    # x = GradVariable(randn(m, n))
    # y = tanh(CallbackStack(), x)
    # @test isa(y, GradVariable)
    # @test size(y) == (m, n)
    # @test all(tanh(x.data) .== y.data)

    # x = GradVariable(randn(m, n))
    # test_op_grad_mse(tanh, x, wrt=x)
end
test_tanh()