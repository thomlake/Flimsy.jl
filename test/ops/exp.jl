using Flimsy

facts("exp") do
    for (m, n) in [(3, 1), (5, 8)]
        context("$(m)x$(n)") do
            context("Constant") do
                x = Constant(randn(m, n))
                y = exp(RunScope(), x)
                @fact isa(y, Constant) --> true
                @fact size(y) --> (m, n)
                for i in eachindex(y)
                    @fact y.data[i] --> roughly(exp(x.data[i]))
                end

                x = Constant(randn(m, n))
                y = exp(GradScope(), x)
                @fact isa(y, Constant) --> true
                @fact size(y) --> (m, n)
                for i in eachindex(y)
                    @fact y.data[i] --> roughly(exp(x.data[i]))
                end
            end

            context("Variable") do
                x = Variable(randn(m, n))
                y = exp(RunScope(), x)
                @fact isa(y, Constant) --> true
                @fact size(y) --> (m, n)
                for i in eachindex(y)
                    @fact y.data[i] --> roughly(exp(x.data[i]))
                end

                x = Variable(randn(m, n), zeros(m, n))
                y = exp(GradScope(), x)
                @fact isa(y, Variable) --> true
                @fact size(y) --> (m, n)
                for i in eachindex(y)
                    @fact y.data[i] --> roughly(exp(x.data[i]))
                end
            end

            context("Gradient") do
                x = Variable(randn(m, n))
                test_op_grad_mse(exp, x, wrt=x)
            end
        end
    end
end
