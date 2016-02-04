using Flimsy

facts("mult") do
    for (m, n) in [(3, 1), (4, 5)]
        context("$(m)x$(n)") do
            for atype in [DataVariable, GradVariable]
                for btype in [DataVariable, GradVariable]
                    ctxstr = string(
                        atype <: DataVariable ? "DataVariable" : "GradVariable",
                        ",", 
                        btype <: DataVariable ? "DataVariable" : "GradVariable",
                    )
                    context(ctxstr) do
                        context("MxN + MxN") do
                            scope = DynamicScope()
                            gscope = GradScope(scope)

                            a = atype <: DataVariable ? atype(randn(m, n)) : atype(randn(m, n), zeros(m, n))
                            b = btype <: DataVariable ? btype(randn(m, n)) : btype(randn(m, n), zeros(m, n))
                            c = mult(scope, a, b)
                            @fact isa(c, DataVariable) --> true
                            @fact size(c)              --> (m, n)
                            @fact c.data               --> roughly(a.data .* b.data)

                            a = atype <: DataVariable ? atype(randn(m, n)) : atype(randn(m, n), zeros(m, n))
                            b = btype <: DataVariable ? btype(randn(m, n)) : btype(randn(m, n), zeros(m, n))
                            c = mult(gscope, a, b)
                            if anygrads(atype, btype)
                                @fact isa(c, GradVariable) --> true
                            else
                                @fact isa(c, DataVariable) --> true
                            end
                            @fact size(c) --> (m, n)
                            @fact c.data  --> roughly(a.data .* b.data)

                            if anygrads(atype, btype)
                                context("Gradient") do
                                    a = atype <: DataVariable ? atype(randn(m, n)) : atype(randn(m, n), zeros(m, n))
                                    b = btype <: DataVariable ? btype(randn(m, n)) : btype(randn(m, n), zeros(m, n))
                                    wrt = []
                                    atype <: GradVariable && push!(wrt, a)
                                    btype <: GradVariable && push!(wrt, b)
                                    test_op_grad_mse(mult, a, b, wrt=wrt)
                                end
                            end
                        end

                        context("1xN + MxN") do
                            scope = DynamicScope()
                            gscope = GradScope(scope)

                            a = atype <: DataVariable ? atype(randn(1, n)) : atype(randn(1, n), zeros(1, n))
                            b = btype <: DataVariable ? btype(randn(m, n)) : btype(randn(m, n), zeros(m, n))
                            c = mult(scope, a, b)
                            @fact isa(c, DataVariable) --> true
                            @fact size(c)              --> (m, n)
                            @fact c.data               --> roughly(a.data .* b.data)

                            a = atype <: DataVariable ? atype(randn(1, n)) : atype(randn(1, n), zeros(1, n))
                            b = btype <: DataVariable ? btype(randn(m, n)) : btype(randn(m, n), zeros(m, n))
                            c = mult(gscope, a, b)
                            if anygrads(atype, btype)
                                @fact isa(c, GradVariable) --> true
                            else
                                @fact isa(c, DataVariable) --> true
                            end
                            @fact size(c) --> (m, n)
                            @fact c.data  --> roughly(a.data .* b.data)

                            if anygrads(atype, btype)
                                context("Gradient") do
                                    a = atype <: DataVariable ? atype(randn(1, n)) : atype(randn(1, n), zeros(1, n))
                                    b = btype <: DataVariable ? btype(randn(m, n)) : btype(randn(m, n), zeros(m, n))
                                    wrt = []
                                    atype <: GradVariable && push!(wrt, a)
                                    btype <: GradVariable && push!(wrt, b)
                                    test_op_grad_mse(mult, a, b, wrt=wrt)
                                end
                            end
                        end

                        context("MxN + 1xN") do
                            scope = DynamicScope()
                            gscope = GradScope(scope)

                            a = atype <: DataVariable ? atype(randn(m, n)) : atype(randn(m, n), zeros(m, n))
                            b = btype <: DataVariable ? btype(randn(1, n)) : btype(randn(1, n), zeros(1, n))
                            c = mult(scope, a, b)
                            @fact isa(c, DataVariable) --> true
                            @fact size(c)              --> (m, n)
                            @fact c.data               --> roughly(a.data .* b.data)

                            a = atype <: DataVariable ? atype(randn(m, n)) : atype(randn(m, n), zeros(m, n))
                            b = btype <: DataVariable ? btype(randn(1, n)) : btype(randn(1, n), zeros(1, n))
                            c = mult(gscope, a, b)
                            if anygrads(atype, btype)
                                @fact isa(c, GradVariable) --> true
                            else
                                @fact isa(c, DataVariable) --> true
                            end
                            @fact size(c) --> (m, n)
                            @fact c.data  --> roughly(a.data .* b.data)

                            if anygrads(atype, btype)
                                context("Gradient") do
                                    a = atype <: DataVariable ? atype(randn(m, n)) : atype(randn(m, n), zeros(m, n))
                                    b = btype <: DataVariable ? btype(randn(1, n)) : btype(randn(1, n), zeros(1, n))
                                    wrt = []
                                    atype <: GradVariable && push!(wrt, a)
                                    btype <: GradVariable && push!(wrt, b)
                                    test_op_grad_mse(mult, a, b, wrt=wrt)
                                end
                            end
                        end

                        context("Mx1 + MxN") do
                            scope = DynamicScope()
                            gscope = GradScope(scope)

                            a = atype <: DataVariable ? atype(randn(m, 1)) : atype(randn(m, 1), zeros(m, 1))
                            b = btype <: DataVariable ? btype(randn(m, n)) : btype(randn(m, n), zeros(m, n))
                            c = mult(scope, a, b)
                            @fact isa(c, DataVariable) --> true
                            @fact size(c)              --> (m, n)
                            @fact c.data               --> roughly(a.data .* b.data)

                            a = atype <: DataVariable ? atype(randn(m, 1)) : atype(randn(m, 1), zeros(m, 1))
                            b = btype <: DataVariable ? btype(randn(m, n)) : btype(randn(m, n), zeros(m, n))
                            c = mult(gscope, a, b)
                            if anygrads(atype, btype)
                                @fact isa(c, GradVariable) --> true
                            else
                                @fact isa(c, DataVariable) --> true
                            end
                            @fact size(c) --> (m, n)
                            @fact c.data  --> roughly(a.data .* b.data)

                            if anygrads(atype, btype)
                                context("Gradient") do
                                    a = atype <: DataVariable ? atype(randn(m, 1)) : atype(randn(m, 1), zeros(m, 1))
                                    b = btype <: DataVariable ? btype(randn(m, n)) : btype(randn(m, n), zeros(m, n))
                                    wrt = []
                                    atype <: GradVariable && push!(wrt, a)
                                    btype <: GradVariable && push!(wrt, b)
                                    test_op_grad_mse(mult, a, b, wrt=wrt)
                                end
                            end
                        end

                        context("MxN + Mx1") do
                            scope = DynamicScope()
                            gscope = GradScope(scope)

                            a = atype <: DataVariable ? atype(randn(m, n)) : atype(randn(m, n), zeros(m, n))
                            b = btype <: DataVariable ? btype(randn(m, 1)) : btype(randn(m, 1), zeros(m, 1))
                            c = mult(scope, a, b)
                            @fact isa(c, DataVariable) --> true
                            @fact size(c)              --> (m, n)
                            @fact c.data               --> roughly(a.data .* b.data)

                            a = atype <: DataVariable ? atype(randn(m, n)) : atype(randn(m, n), zeros(m, n))
                            b = btype <: DataVariable ? btype(randn(m, 1)) : btype(randn(m, 1), zeros(m, 1))
                            c = mult(gscope, a, b)
                            if anygrads(atype, btype)
                                @fact isa(c, GradVariable) --> true
                            else
                                @fact isa(c, DataVariable) --> true
                            end
                            @fact size(c) --> (m, n)
                            @fact c.data  --> roughly(a.data .* b.data)

                            if anygrads(atype, btype)
                                context("Gradient") do
                                    a = atype <: DataVariable ? atype(randn(m, n)) : atype(randn(m, n), zeros(m, n))
                                    b = btype <: DataVariable ? btype(randn(m, 1)) : btype(randn(m, 1), zeros(m, 1))
                                    wrt = []
                                    atype <: GradVariable && push!(wrt, a)
                                    btype <: GradVariable && push!(wrt, b)
                                    test_op_grad_mse(mult, a, b, wrt=wrt)
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
