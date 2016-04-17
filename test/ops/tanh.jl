
facts("tanh") do
    for (m, n) in [(3, 1), (5, 8)]
        context("$(m)x$(n)") do
            
            context("Scope") do
                scope = DataScope()
                x = DataVariable(randn(m, n))
                y = tanh(scope, x)
                @fact isa(y, DataVariable) --> true
                @fact size(y) --> (m, n) 
                @fact y.data --> roughly(tanh(x.data))

                x = GradVariable(randn(m, n), zeros(m, n))
                y = tanh(scope, x)
                @fact isa(y, DataVariable) --> true
                @fact size(y) --> (m, n)
                @fact y.data --> roughly(tanh(x.data))
            end

            context("GradScope") do
                scope = GradScope()
                x = DataVariable(randn(m, n))
                y = tanh(scope, x)
                @fact isa(y, DataVariable) --> true
                @fact size(y) --> (m, n)
                @fact y.data --> roughly(tanh(x.data))

                x = GradVariable(randn(m, n), zeros(m, n))
                y = tanh(scope, x)
                @fact isa(y, GradVariable) --> true
                @fact size(y) --> (m, n)
                @fact y.data --> roughly(tanh(x.data))
            end

            context("Gradient") do
                x = GradVariable(randn(m, n), zeros(m, n))
                test_op_grad_mse(tanh, x, wrt=x)
            end
        end
    end
end
