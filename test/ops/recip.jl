using Flimsy

facts("recip") do
    for (m, n) in [(3, 1), (5, 8)]
        context("$(m)x$(n)") do
            context("DataVariable") do
                scope = DynamicScope()
                gscope = GradScope(scope)

                x = DataVariable(randn(m, n))
                y = recip(scope, x)
                @fact isa(y, DataVariable) --> true
                @fact size(y)              --> (m, n)
                @fact y.data               --> roughly(1 ./ x.data)

                x = DataVariable(randn(m, n))
                y = recip(gscope, x)
                @fact isa(y, DataVariable) --> true
                @fact size(y)              --> (m, n)
                @fact y.data               --> roughly(1 ./ x.data)
            end

            context("GradVariable") do
                scope = DynamicScope()
                gscope = GradScope(scope)

                x = GradVariable(randn(m, n), zeros(m, n))
                y = recip(scope, x)
                @fact isa(y, DataVariable) --> true
                @fact size(y)              --> (m, n)
                @fact y.data               --> roughly(1 ./ x.data)

                x = GradVariable(randn(m, n), zeros(m, n))
                y = recip(gscope, x)
                @fact isa(y, GradVariable) --> true
                @fact size(y)              --> (m, n)
                @fact y.data               --> roughly(1 ./ x.data)
            end

            context("Gradient") do
                x = GradVariable(randn(m, n), zeros(m, n))
                test_op_grad_mse(recip, x, wrt=x)
            end
        end
    end
end
