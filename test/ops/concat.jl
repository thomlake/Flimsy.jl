using Flimsy

facts("concat") do
    m_min, m_max = 2, 7
    for K in [3, 4, 5]
        for n in [1, 3]
            context("$(K)xMx$(n)") do
                for typ in [DataVariable, GradVariable]
                    context(typ <: DataVariable ? "DataVariable" : "GradVariable") do
                        scope = DynamicScope()
                        gradscope = GradScope(scope)

                        ms = rand(m_min:m_max, K)
                        m = sum(ms)
                        xs = typ[typ(randn(ms[k], n)) for k = 1:K]
                        y = concat(scope, xs)
                        @fact isa(y, DataVariable) --> true
                        @fact size(y)              --> (m, n)
                        @fact y.data               --> vcat([x.data for x in xs]...)

                        y = concat(gradscope, xs)
                        if typ <: DataVariable
                            @fact isa(y, DataVariable) --> true
                            @fact isa(y, GradVariable) --> false
                        else
                            @fact isa(y, DataVariable) --> false
                            @fact isa(y, GradVariable) --> true
                        end
                        @fact size(y)                  --> (m, n)
                        @fact y.data                   --> vcat([x.data for x in xs]...)

                    end
                end
                
                context("Gradient") do
                    ms = rand(m_min:m_max, K)
                    xs = [GradVariable(randn(ms[k], n)) for k = 1:K]
                    test_op_grad_mse(concat, xs, wrt=xs)
                end
            end
        end
    end
end
