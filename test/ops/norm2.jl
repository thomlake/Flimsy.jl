using Flimsy

facts("norm2") do
    for (m, n) in [(10, 1), (5, 8)]
        context("$(m)x$(n)") do
            context("Constant") do
                x = Constant(randn(m, n))
                y = norm2(RunScope(), x)
                @fact isa(y, Constant) --> true
                @fact size(y)          --> (1, n)
                @fact y.data           --> roughly(mapslices(norm, x.data, 1))

                x = Constant(randn(m, n))
                y = norm2(GradScope(), x)
                @fact isa(y, Constant) --> true
                @fact size(y)          --> (1, n)
                @fact y.data           --> roughly(mapslices(norm, x.data, 1))
            end

            context("Variable") do
                x = Variable(randn(m, n))
                y = norm2(RunScope(), x)
                @fact isa(y, Constant) --> true
                @fact size(y)          --> (1, n)
                @fact y.data           --> roughly(mapslices(norm, x.data, 1))

                x = Variable(randn(m, n))
                y = norm2(GradScope(), x)
                @fact isa(y, Variable) --> true
                @fact size(y)          --> (1, n)
                @fact y.data           --> roughly(mapslices(norm, x.data, 1))
            end

            context("Gradient") do
                x = Variable(10 * randn(m, n), zeros(m, n))
                test_op_grad_mse(norm2, x, wrt=x)
            end
        end
    end
end
