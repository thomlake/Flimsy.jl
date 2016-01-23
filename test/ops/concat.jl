using Flimsy
using Base.Test


function test_concat()
    m_min, m_max = 2, 7
    for K in [3, 4, 5]
        for n in [1, 3]
            for typ in [DataVariable, GradVariable]
                ms = rand(m_min:m_max, K)
                m = sum(ms)
                xs = typ[typ(randn(ms[k], n)) for k = 1:K]
                y = concat(xs)
                @test isa(y, DataVariable)
                @test size(y) == (m, n)
                @test all(vcat([x.data for x in xs]...) .== y.data)

                if typ <: GradVariable
                    ms = rand(m_min:m_max, K)
                    xs = typ[typ(randn(ms[k], n)) for k = 1:K]
                    test_op_grad_mse(concat, xs, wrt=xs)
                end
            end
        end
    end
end
test_concat()
