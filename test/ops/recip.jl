using Flimsy

facts("recip") do
    for (m, n) in [(3, 1), (5, 8)]
        context("$(m)x$(n)") do
            context("Constant") do
                rscope = RunScope()
                gscope = GradScope()

                x = Constant(randn(m, n))
                y = recip(rscope, x)
                @fact isa(y, Constant) --> true
                @fact size(y) --> (m, n)
                @fact y.data --> roughly(1 ./ x.data)

                x = Constant(randn(m, n))
                y = recip(gscope, x)
                @fact isa(y, Constant) --> true
                @fact size(y) --> (m, n)
                @fact y.data --> roughly(1 ./ x.data)
            end

            context("Variable") do
                rscope = RunScope()
                gscope = GradScope()

                x = Variable(randn(m, n), zeros(m, n))
                y = recip(rscope, x)
                @fact isa(y, Constant) --> true
                @fact size(y) --> (m, n)
                @fact y.data --> roughly(1 ./ x.data)

                x = Variable(randn(m, n), zeros(m, n))
                y = recip(gscope, x)
                @fact isa(y, Variable) --> true
                @fact size(y) --> (m, n)
                @fact y.data --> roughly(1 ./ x.data)
            end

            context("Gradient") do
                x = Variable(randn(m, n), zeros(m, n))
                test_op_grad_mse(recip, x, wrt=x)
            end
        end
    end
end
