using Flimsy

facts("affine") do 
    for (m, n, k) in [(4, 3, 1), (5, 10, 7)]
        context("$(m)x$(n)x$(k)") do
            for wtype in [DataVariable, GradVariable]
                for xtype in [DataVariable, GradVariable]
                    for btype in [DataVariable, GradVariable]
                        ctxstr = string(
                            wtype <: DataVariable ? "DataVariable" : "GradVariable",
                            ",", 
                            xtype <: DataVariable ? "DataVariable" : "GradVariable",
                            ",",
                            btype <: DataVariable ? "DataVariable" : "GradVariable",
                        )
                        context(ctxstr) do
                            scope = DataScope()
                            gscope = GradScope()

                            ysz = m, k
                            w = wtype <: DataVariable ? wtype(randn(m, n)) : wtype(randn(m, n), zeros(m, n))
                            x = xtype <: DataVariable ? xtype(randn(n, k)) : xtype(randn(n, k), zeros(n, k))
                            b = btype <: DataVariable ? btype(randn(m, 1)) : btype(randn(m, 1), zeros(m, 1))
                            y = affine(scope, w, x, b)
                            @fact isa(y, DataVariable) --> true
                            @fact size(y)              --> (m, k)
                            @fact y.data               --> roughly(w.data * x.data .+ b.data)

                            w = wtype <: DataVariable ? wtype(randn(m, n)) : wtype(randn(m, n), zeros(m, n))
                            x = xtype <: DataVariable ? xtype(randn(n, k)) : xtype(randn(n, k), zeros(n, k))
                            b = btype <: DataVariable ? btype(randn(m, 1)) : btype(randn(m, 1), zeros(m, 1))
                            y = affine(gscope, w, x, b)
                            if anygrads(wtype, xtype, btype)
                                @fact isa(y, GradVariable) --> true
                            else
                                @fact isa(y, DataVariable) --> true
                            end
                            @fact size(y) --> (m, k)
                            @fact y.data  --> roughly(w.data * x.data .+ b.data)

                            if anygrads(wtype, xtype, btype)
                                w = wtype <: DataVariable ? wtype(randn(m, n)) : wtype(randn(m, n), zeros(m, n))
                                x = xtype <: DataVariable ? xtype(randn(n, k)) : xtype(randn(n, k), zeros(n, k))
                                b = btype <: DataVariable ? btype(randn(m, 1)) : btype(randn(m, 1), zeros(m, 1))
                                wrt = []
                                isa(w, GradVariable) && push!(wrt, w)
                                isa(x, GradVariable) && push!(wrt, x)
                                isa(b, GradVariable) && push!(wrt, b) 
                                test_op_grad_mse(affine, w, x, b, wrt=wrt)
                            end
                        end
                    end
                end
            end
        end
    end
end
