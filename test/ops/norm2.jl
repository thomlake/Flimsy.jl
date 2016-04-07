using Flimsy

facts("norm2") do
    for (m, n) in [(10, 1), (5, 8)]
        context("$(m)x$(n)") do
            context("DataVariable") do
                scope = DynamicScope()
                gscope = GradScope(scope)

                x = DataVariable(randn(m, n))
                y = norm2(scope, x)
                @fact isa(y, DataVariable) --> true
                @fact size(y)              --> (1, n)
                @fact y.data               --> roughly(mapslices(norm, x.data, 1))

                x = DataVariable(randn(m, n))
                y = norm2(gscope, x)
                @fact isa(y, DataVariable) --> true
                @fact size(y)              --> (1, n)
                @fact y.data               --> roughly(mapslices(norm, x.data, 1))
            end

            context("GradVariable") do
                scope = DynamicScope()
                gscope = GradScope(scope)

                x = GradVariable(randn(m, n), zeros(m, n))
                y = norm2(scope, x)
                @fact isa(y, DataVariable) --> true
                @fact size(y)              --> (1, n)
                @fact y.data               --> roughly(mapslices(norm, x.data, 1))

                x = GradVariable(randn(m, n), zeros(m, n))
                y = norm2(gscope, x)
                @fact isa(y, GradVariable) --> true
                @fact size(y)              --> (1, n)
                @fact y.data               --> roughly(mapslices(norm, x.data, 1))
            end

            context("Gradient") do
                x = GradVariable(10 * randn(m, n), zeros(m, n))
                test_op_grad_mse(norm2, x, wrt=x)
            end
        end
    end
end
