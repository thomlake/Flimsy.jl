using Flimsy

facts("mult_scalar") do
    for (m, n) in [(3, 1), (5, 8)]
        context("$(m)x$(n)") do
            for a in [1.5, 0.5, -0.5, 1.5]
                for typ in [DataVariable, GradVariable]
                    scope = DynamicScope()
                    gscope = GradScope(scope)

                    x = typ <: DataVariable ? typ(randn(m, n)) : typ(randn(m, n), zeros(m, n))
                    y = mult(scope, a, x)
                    @fact isa(y, DataVariable) --> true
                    @fact size(y)              --> (m, n)
                    @fact y.data               --> roughly(a .* x.data)

                    x = typ <: DataVariable ? typ(randn(m, n)) : typ(randn(m, n), zeros(m, n))
                    y = mult(gscope, a, x)
                    if typ <: GradVariable
                        @fact isa(y, GradVariable) --> true
                    else
                        @fact isa(y, DataVariable) --> true
                    end
                    @fact size(y) --> (m, n)
                    @fact y.data  --> roughly(a .* x.data)

                    
                    x = typ <: DataVariable ? typ(randn(m, n)) : typ(randn(m, n), zeros(m, n))
                    y = mult(scope, x, a)
                    @fact isa(y, DataVariable) --> true
                    @fact size(y)              --> (m, n)
                    @fact y.data               --> roughly(a .* x.data)

                    x = typ <: DataVariable ? typ(randn(m, n)) : typ(randn(m, n), zeros(m, n))
                    y = mult(gscope, x, a)
                    if typ <: GradVariable
                        @fact isa(y, GradVariable) --> true
                    else
                        @fact isa(y, DataVariable) --> true
                    end
                    @fact size(y) --> (m, n)
                    @fact y.data  --> roughly(a .* x.data)

                end

                x = GradVariable(randn(m, n), zeros(m, n))
                test_op_grad_mse(mult, a, x, wrt=x)

                x = GradVariable(randn(m, n), zeros(m, n))
                test_op_grad_mse(mult, x, a, wrt=x)
            end
        end
    end
end
