using Flimsy

facts("minus(Real, Variable)") do
    for (m, n) in [(3, 1), (5, 8)]
        context("$(m)x$(n)") do
            for typ in [DataVariable, GradVariable]
                context(typ <: DataVariable ? "DataVariable" : "GradVariable") do
                    for a in [1.5, 0.5, -0.5, -1.5]
                        scope = DynamicScope()
                        gscope = GradScope(scope)

                        x = typ <: DataVariable ? DataVariable(randn(m, n)) : GradVariable(randn(m, n), zeros(m, n))
                        y = minus(scope, a, x)
                        @fact isa(y, DataVariable) --> true
                        @fact size(y)              --> (m, n)
                        @fact y.data               --> a .- x.data 

                        x = typ <: DataVariable ? DataVariable(randn(m, n)) : GradVariable(randn(m, n), zeros(m, n))
                        y = minus(gscope, a, x)
                        if typ <: GradVariable
                            @fact isa(y, GradVariable) --> true
                        else
                            @fact isa(y, DataVariable) --> true
                        end
                        @fact size(y) --> (m, n)
                        @fact y.data  --> a .- x.data                                                 
                    end
                end
            end

            context("Gradient") do
                for a in [1.5, 0.5, -0.5, -1.5]
                    x = GradVariable(randn(m, n), zeros(m, n))
                    test_op_grad_mse(minus, a, x, wrt=x)                                                
                end
            end
        end
    end
end

facts("minus(Variable, Real)") do
    for (m, n) in [(3, 1), (5, 8)]
        context("$(m)x$(n)") do
            for typ in [DataVariable, GradVariable]
                context(typ <: DataVariable ? "DataVariable" : "GradVariable") do
                    for a in [1.5, 0.5, -0.5, -1.5]
                        scope = DynamicScope()
                        gscope = GradScope(scope)

                        x = typ <: DataVariable ? DataVariable(randn(m, n)) : GradVariable(randn(m, n), zeros(m, n))
                        y = minus(scope, x, a)
                        @fact isa(y, DataVariable) --> true
                        @fact size(y)              --> (m, n)
                        @fact y.data               --> x.data .- a

                        x = typ <: DataVariable ? DataVariable(randn(m, n)) : GradVariable(randn(m, n), zeros(m, n))
                        y = minus(gscope, x, a)
                        if typ <: GradVariable
                            @fact isa(y, GradVariable) --> true
                        else
                            @fact isa(y, DataVariable) --> true
                        end
                        @fact size(y) --> (m, n)
                        @fact y.data  --> x.data .- a
                    end
                end
            end

            context("Gradient") do
                for a in [1.5, 0.5, -0.5, -1.5]
                    x = GradVariable(randn(m, n), zeros(m, n))
                    test_op_grad_mse(minus, x, a, wrt=x)                                                
                end
            end
        end
    end
end
