using Flimsy
using Base.Test

function test_decat()
    f(x) = concat(decat(x))
    f(s, x) = concat(s, decat(s, x))

    for (m, n) in [(6, 1), (3, 9)]
        for typ in [DataVariable, GradVariable]
            x = typ(randn(m, n))
            y = decat(x)
            @test isa(y, Vector)
            @test eltype(y) <: DataVariable
            for i = 1:m
                @test size(y[i]) == (1, n)
                for j = 1:n
                    @test_approx_eq x.data[i,j] y[i].data[j]
                end
            end

            x = typ(randn(m, n))
            y = decat(CallbackStack(), x)
            @test isa(y, Vector)
            @test eltype(y) <: (typ <: GradVariable ? GradVariable : DataVariable)
            for i = 1:m
                @test size(y[i]) == (1, n)
                for j = 1:n
                    @test_approx_eq x.data[i,j] y[i].data[j]
                end
            end

            if typ <: GradVariable
                x = typ(randn(m, n))
                test_op_grad_mse(f, x, wrt=x)
            end
        end
    end
end
test_decat()
