using Flimsy

facts("identity") do
    for (m, n) in [(1, 3), (3, 1), (5, 6)]
        context("$(m)x$(n)") do
            rscope = RunScope()
            gscope = GradScope()
            x = Constant(randn(m, n))
            y = identity(rscope, x)
            @fact isa(y, Constant) --> true
            @fact size(y) --> (m, n)
            @fact y.data --> x.data 

            y = identity(gscope, x)
            @fact isa(y, Constant) --> true
            @fact size(y) --> (m, n)
            @fact y.data --> x.data 

            x = Variable(randn(m, n), zeros(m, n))
            y = identity(rscope, x)
            @fact isa(y, Constant) --> true
            @fact size(y) --> (m, n)
            @fact y.data --> x.data

            y = identity(gscope, x)
            @fact isa(y, Variable) --> true
            @fact size(y) --> (m, n)
            @fact y.data --> x.data

            x = Variable(randn(m, n), zeros(m, n))
            test_op_grad_mse(identity, x, wrt=x)
        end
    end
end
