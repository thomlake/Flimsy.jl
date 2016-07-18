
facts("tanh") do
    for (m, n) in [(3, 1), (5, 8)]
        context("$(m)x$(n)") do
            
            context("RunScope") do
                scope = RunScope()
                x = Constant(randn(m, n))
                y = tanh(scope, x)
                @fact isa(y, Constant) --> true
                @fact size(y) --> (m, n) 
                @fact y.data --> roughly(tanh(x.data))

                x = Variable(randn(m, n))
                y = tanh(scope, x)
                @fact isa(y, Constant) --> true
                @fact size(y) --> (m, n)
                @fact y.data --> roughly(tanh(x.data))
            end

            context("GradScope") do
                scope = GradScope()
                x = Constant(randn(m, n))
                y = tanh(scope, x)
                @fact isa(y, Constant) --> true
                @fact size(y) --> (m, n)
                @fact y.data --> roughly(tanh(x.data))

                x = Variable(randn(m, n))
                y = tanh(scope, x)
                @fact isa(y, Variable) --> true
                @fact size(y) --> (m, n)
                @fact y.data --> roughly(tanh(x.data))
            end

            context("Gradient") do
                x = Variable(randn(m, n))
                test_op_grad_mse(tanh, x, wrt=x)
            end
        end
    end
end
