using Flimsy

facts("relu") do
    for (m, n) in [(3, 1), (5, 8)]
        context("$(m)x$(n)") do
            context("DataVariable") do
                scope = DataScope()
                gscope = GradScope()

                x = DataVariable(randn(m, n))
                y = relu(scope, x)
                @fact isa(y, DataVariable) --> true
                @fact size(y)              --> (m, n)
                @fact y.data               --> roughly(relu(x.data))

                x = DataVariable(randn(m, n))
                y = relu(gscope, x)
                @fact isa(y, DataVariable) --> true
                @fact size(y)              --> (m, n)
                @fact y.data               --> roughly(relu(x.data))
            end

            context("GradVariable") do
                scope = DataScope()
                gscope = GradScope()

                x = GradVariable(randn(m, n), zeros(m, n))
                y = relu(scope, x)
                @fact isa(y, DataVariable) --> true
                @fact size(y)              --> (m, n)
                @fact y.data               --> roughly(relu(x.data))

                x = GradVariable(randn(m, n), zeros(m, n))
                y = relu(gscope, x)
                @fact isa(y, GradVariable) --> true
                @fact size(y)              --> (m, n)
                @fact y.data               --> roughly(relu(x.data))
            end

            context("Gradient") do
                x = GradVariable(randn(m, n), zeros(m, n))
                test_op_grad_mse(relu, x, wrt=x)
            end
        end
    end
end
