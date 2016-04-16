using Flimsy

facts("log") do
    for (m, n) in [(3, 1), (5, 8)]
        context("$(m)x$(n)") do
            context("DataVariable") do
                scope = DynamicScope()
                gscope = GradScope(scope)

                x = DataVariable(rand(m, n))
                y = log(scope, x)
                @fact isa(y, DataVariable) --> true
                @fact size(y)              --> (m, n)
                @fact y.data               --> roughly(log(x.data))

                x = DataVariable(rand(m, n))
                y = log(gscope, x)
                @fact isa(y, DataVariable) --> true
                @fact size(y)              --> (m, n)
                @fact y.data               --> roughly(log(x.data))
            end

            context("GradVariable") do
                scope = DynamicScope()
                gscope = GradScope(scope)

                x = GradVariable(rand(m, n), zeros(m, n))
                y = log(scope, x)
                @fact isa(y, DataVariable) --> true
                @fact size(y)              --> (m, n)
                @fact y.data               --> roughly(log(x.data))

                x = GradVariable(rand(m, n), zeros(m, n))
                y = log(gscope, x)
                @fact isa(y, GradVariable) --> true
                @fact size(y)              --> (m, n)
                @fact y.data               --> roughly(log(x.data))
            end

            context("Gradient") do
                x = GradVariable(rand(m, n), zeros(m, n))
                test_op_grad_mse(log, x, wrt=x)
            end
        end
    end
end
