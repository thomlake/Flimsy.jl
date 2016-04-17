using Flimsy

facts("wta") do
    for (m, n) in [(3, 1), (5, 8)]
        context("$(m)x$(n)") do
            context("DataVariable") do
                scope = DataScope()
                gscope = GradScope()

                x = DataVariable(randn(m, n))
                y = wta(scope, x)
                @fact isa(y, DataVariable) --> true
                @fact size(y)              --> (m, n)
                @fact y.data               --> roughly(wta(x.data))

                x = DataVariable(randn(m, n))
                y = wta(gscope, x)
                @fact isa(y, DataVariable) --> true
                @fact size(y)              --> (m, n)
                @fact y.data               --> roughly(wta(x.data))
            end

            context("GradVariable") do
                scope = DataScope()
                gscope = GradScope()

                x = GradVariable(randn(m, n), zeros(m, n))
                y = wta(scope, x)
                @fact isa(y, DataVariable) --> true
                @fact size(y)              --> (m, n)
                @fact y.data               --> roughly(wta(x.data))

                x = GradVariable(randn(m, n), zeros(m, n))
                y = wta(gscope, x)
                @fact isa(y, GradVariable) --> true
                @fact size(y)              --> (m, n)
                @fact y.data               --> roughly(wta(x.data))
            end

            context("Gradient") do
                x = GradVariable(randn(m, n), zeros(m, n))
                test_op_grad_mse(wta, x, wrt=x)
            end
        end
    end
end
