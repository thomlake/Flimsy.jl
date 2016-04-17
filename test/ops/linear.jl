using Flimsy

facts("linear") do
    for (wsz, xsz) in [((1, 4), (4, 1)), ((4, 3), (3, 1)), ((1, 4), (4, 5)), ((3, 6), (6, 7))]
        context(string(join(wsz, "x"), " * ", join(xsz, "x"))) do
            for wtype in [DataVariable, GradVariable]
                for xtype in [DataVariable, GradVariable]
                    ctxstr = string(
                        wtype <: DataVariable ? "DataVariable" : "GradVariable",
                        ",", 
                        xtype <: DataVariable ? "DataVariable" : "GradVariable",
                    )
                    context(ctxstr) do
                        scope = DataScope()
                        gscope = GradScope()

                        ysz = (wsz[1], xsz[2])
                        w = wtype <: DataVariable ? wtype(randn(wsz)) : wtype(randn(wsz), zeros(wsz))
                        x = xtype <: DataVariable ? xtype(randn(xsz)) : xtype(randn(xsz), zeros(xsz))
                        y = linear(scope, w, x)
                        @fact isa(y, DataVariable) --> true
                        @fact size(y)              --> ysz
                        @fact y.data               --> w.data * x.data

                        w = wtype <: DataVariable ? wtype(randn(wsz)) : wtype(randn(wsz), zeros(wsz))
                        x = xtype <: DataVariable ? xtype(randn(xsz)) : xtype(randn(xsz), zeros(xsz))
                        y = linear(gscope, w, x)
                        target_type = anygrads(wtype, xtype) ? GradVariable : DataVariable 
                        @fact isa(y, target_type) --> true
                        @fact size(y)             --> ysz
                        @fact y.data              --> w.data * x.data

                        if anygrads(wtype, xtype)
                            context("Gradient") do
                                w = wtype <: DataVariable ? wtype(randn(wsz)) : wtype(randn(wsz), zeros(wsz))
                                x = xtype <: DataVariable ? xtype(randn(xsz)) : xtype(randn(xsz), zeros(xsz))
                                wrt = []
                                wtype <: GradVariable && push!(wrt, w)
                                xtype <: GradVariable && push!(wrt, x)
                                test_op_grad_mse(linear, w, x, wrt=wrt)
                            end
                        end
                    end
                end
            end
        end
    end
end
