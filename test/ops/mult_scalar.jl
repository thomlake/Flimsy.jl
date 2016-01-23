using Flimsy
using Base.Test

function test_mult_scalar()
    for (m, n) in [(3, 1), (5, 8)]
        for a in [1.5, 0.5, -0.5, 1.5]
            for t in [DataVariable, GradVariable]
                # mult(Real, MxN)
                x = t(n == 1 ? randn(m) : randn(m, n))
                y = mult(a, x)
                @test isa(y, DataVariable)
                @test size(y) == (m, n)
                @test_approx_eq a .* x.data y.data

                x = t(n == 1 ? randn(m) : randn(m, n))
                y = mult(CallbackStack(), a, x)
                @test isa(y, t)
                @test size(y) == (m, n)
                @test_approx_eq a .* x.data y.data

                # mult(MxN, Real)
                x = t(n == 1 ? randn(m) : randn(m, n))
                y = mult(x, a)
                @test isa(y, DataVariable)
                @test size(y) == (m, n)
                @test_approx_eq a .* x.data y.data

                x = t(n == 1 ? randn(m) : randn(m, n))
                y = mult(CallbackStack(), x, a)
                @test isa(y, t)
                @test size(y) == (m, n)
                @test_approx_eq a .* x.data y.data
            end

            x = GradVariable(n == 1 ? randn(m) : randn(m, n))
            test_op_grad_mse(mult, a, x, wrt=x)

            x = GradVariable(n == 1 ? randn(m) : randn(m, n))
            test_op_grad_mse(mult, x, a, wrt=x)
        end
    end
end
test_mult_scalar()
