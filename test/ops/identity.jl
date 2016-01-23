using Flimsy
using Base.Test

function test_identity()
    for sz in [(1, 3), (3, 1), (5, 6)]
        for typ in [DataVariable, GradVariable]
            x = typ(randn(sz))
            y = identity(x)
            @test isa(y, typ)
            @test size(y) == sz
            @test_approx_eq x.data y.data

            if typ <: GradVariable
                x = typ(randn(sz))
                test_op_grad_mse(identity, x, wrt=x)
            end
        end
    end
end
test_identity()
