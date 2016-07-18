using Flimsy

facts("log") do
    for (m, n) in [(3, 1), (5, 8)]
        context("$(m)x$(n)") do
            context("Constant") do
                x = Constant(rand(m, n))
                y = log(RunScope(), x)
                @fact isa(y, Constant) --> true
                @fact size(y) --> (m, n)
                @fact y.data --> roughly(log(x.data))

                x = Constant(rand(m, n))
                y = log(GradScope(), x)
                @fact isa(y, Constant) --> true
                @fact size(y) --> (m, n)
                @fact y.data --> roughly(log(x.data))
            end

            context("Variable") do
                x = Variable(rand(m, n))
                y = log(RunScope(), x)
                @fact isa(y, Constant) --> true
                @fact size(y) --> (m, n)
                @fact y.data --> roughly(log(x.data))

                x = Variable(rand(m, n), zeros(m, n))
                y = log(GradScope(), x)
                @fact isa(y, Variable) --> true
                @fact size(y) --> (m, n)
                @fact y.data --> roughly(log(x.data))
            end

            context("Gradient") do
                x = Variable(rand(m, n))
                test_op_grad_mse(log, x; wrt=x)
            end
        end
    end
end
