using Flimsy

facts("sigmoid") do
    for (m, n) in [(3, 1), (5, 8)]
        context("$(m)x$(n)") do
            context("Constant") do
                x = Constant(randn(m, n))
                y = sigmoid(RunScope(), x)
                @fact isa(y, Constant) --> true
                @fact size(y) --> (m, n)
                @fact y.data --> roughly(sigmoid(x.data))

                x = Constant(randn(m, n))
                y = sigmoid(GradScope(), x)
                @fact isa(y, Constant) --> true
                @fact size(y) --> (m, n)
                @fact y.data --> roughly(sigmoid(x.data))
            end

            context("Variable") do
                x = Variable(randn(m, n))
                y = sigmoid(RunScope(), x)
                @fact isa(y, Constant) --> true
                @fact size(y) --> (m, n)
                @fact y.data --> roughly(sigmoid(x.data))

                x = Variable(randn(m, n))
                y = sigmoid(GradScope(), x)
                @fact isa(y, Variable) --> true
                @fact size(y) --> (m, n)
                @fact y.data --> roughly(sigmoid(x.data))
            end

            context("Gradient") do
                x = Variable(randn(m, n))
                test_op_grad_mse(sigmoid, x, wrt=x)
            end
        end
    end
end
