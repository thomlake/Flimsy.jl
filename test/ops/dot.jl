using Flimsy

facts("dot") do
    for (m, n) in [(1, 1), (1, 4), (4, 1), (4, 3), (4, 5)]
        context(string(m, "x", n)) do
            for Ta in [DataVariable, GradVariable]
                for Tb in [DataVariable, GradVariable]
                    context(string(Ta, ", ", Tb)) do
                        a = Ta <: DataVariable ? Ta(randn(m, n)) : Ta(randn(m, n), zeros(m, n))
                        b = Tb <: DataVariable ? Tb(randn(m, n)) : Tb(randn(m, n), zeros(m, n))
                        scope = DataScope()
                        gscope = GradScope()
                        
                        c = dot(scope, a, b)
                        @fact isa(c, DataVariable) --> true
                        @fact size(c) --> (1, n)
                        for j = 1:n
                            @fact c.data[1,j] --> dot(a.data[:,j], b.data[:,j])
                        end

                        c = dot(gscope, a, b)
                        Tc = anygrads(Ta, Tb) ? GradVariable : DataVariable
                        @fact isa(c, Tc) --> true
                        @fact size(c) --> (1, n)
                        for j = 1:n
                            @fact c.data[1,j] --> dot(a.data[:,j], b.data[:,j])
                        end

                        if anygrads(Ta, Tb)
                            context("Gradient") do
                                wrt = []
                                a = Ta <: DataVariable ? Ta(randn(m, n)) : Ta(randn(m, n), zeros(m, n))
                                if isa(a, GradVariable)
                                    push!(wrt, a)
                                end
                                b = Tb <: DataVariable ? Tb(randn(m, n)) : Tb(randn(m, n), zeros(m, n))
                                if isa(b, GradVariable)
                                    push!(wrt, b)
                                end
                                test_op_grad_mse(dot, a, b, wrt=wrt)
                            end
                        end
                    end
                end
            end
        end
    end
end
