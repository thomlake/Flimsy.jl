using Flimsy
using Base.Test

function test_linear()
    for (wsz, xsz) in [((1, 4), (4, 1)), ((4, 3), (3, 1)), ((1, 4), (4, 5)), ((3, 6), (6, 7))]
        for wtype in [DataVariable, GradVariable]
            for xtype in [DataVariable, GradVariable]
                ysz = (wsz[1], xsz[2])

                w = wtype(randn(wsz))
                x = xtype(randn(xsz))
                y = linear(w, x)
                @test isa(y, DataVariable)
                @test size(y) == ysz
                @test_approx_eq w.data * x.data y.data

                w = wtype(randn(wsz))
                x = xtype(randn(xsz))
                y = linear(CallbackStack(), w, x)
                @test isa(y, anygrads(wtype, xtype) ? GradVariable : DataVariable)
                @test size(y) == ysz
                @test_approx_eq w.data * x.data y.data

                if anygrads(wtype, xtype)
                    w = wtype(randn(wsz))
                    x = xtype(randn(xsz))
                    wrt = []
                    wtype <: GradVariable && push!(wrt, w)
                    xtype <: GradVariable && push!(wrt, x)
                    test_op_grad_mse(linear, w, x, wrt=wrt)
                end
            end
        end
    end
end
test_linear()
