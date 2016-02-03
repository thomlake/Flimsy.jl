using Flimsy

facts("sigmoid") do
    for (m, n) in [(3, 1), (5, 8)]
        context("$(m)x$(n)") do
            context("DataVariable") do
                scope = DynamicScope()
                gscope = GradScope(scope)

                x = DataVariable(randn(m, n))
                y = sigmoid(scope, x)
                @fact isa(y, DataVariable) --> true
                @fact size(y)              --> (m, n)
                @fact y.data               --> roughly(sigmoid(x.data))

                x = DataVariable(randn(m, n))
                y = sigmoid(gscope, x)
                @fact isa(y, DataVariable) --> true
                @fact size(y)              --> (m, n)
                @fact y.data               --> roughly(sigmoid(x.data))
            end

            context("GradVariable") do
                scope = DynamicScope()
                gscope = GradScope(scope)

                x = GradVariable(randn(m, n), zeros(m, n))
                y = sigmoid(scope, x)
                @fact isa(y, DataVariable) --> true
                @fact size(y)              --> (m, n)
                @fact y.data               --> roughly(sigmoid(x.data))

                x = GradVariable(randn(m, n), zeros(m, n))
                y = sigmoid(gscope, x)
                @fact isa(y, GradVariable) --> true
                @fact size(y)              --> (m, n)
                @fact y.data               --> roughly(sigmoid(x.data))
            end

            context("Gradient") do
                x = GradVariable(randn(m, n), zeros(m, n))
                test_op_grad_mse(sigmoid, x, wrt=x)
            end
        end
    end
end
