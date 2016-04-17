using Flimsy

facts("identity") do
    for (m, n) in [(1, 3), (3, 1), (5, 6)]
        context("$(m)x$(n)") do
            scope = DataScope()
            x = DataVariable(randn(m, n))
            y = identity(scope, x)
            @fact isa(y, DataVariable) --> true
            @fact size(y)              --> (m, n)
            @fact y.data               --> x.data 

            x = GradVariable(randn(m, n), zeros(m, n))
            y = identity(scope, x)
            @fact isa(y, GradVariable) --> true
            @fact size(y)              --> (m, n)
            @fact y.data               --> x.data

            x = GradVariable(randn(m, n), zeros(m, n))
            test_op_grad_mse(identity, x, wrt=x)
        end
    end
end
