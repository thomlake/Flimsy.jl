using Flimsy
using Base.Test

function test_affine()
    for (m, n, k) in [(4, 3, 1), (5, 10, 7)]
        for wtype in [DataVariable, GradVariable]
            for xtype in [DataVariable, GradVariable]
                for btype in [DataVariable, GradVariable]
                    w, x, b = wtype(randn(m, n)), xtype(randn(n, k)), btype(randn(m))
                    y = affine(w, x, b)
                    @test isa(y, DataVariable)
                    @test size(y) == (m, k)
                    @test_approx_eq w.data * x.data .+ b.data y.data

                    w, x, b = wtype(randn(m, n)), xtype(randn(n, k)), btype(randn(m))
                    y = affine(CallbackStack(), w, x, b)
                    @test isa(y, anygrads(wtype, xtype, btype) ? GradVariable : DataVariable)
                    @test size(y) == (m, k)
                    @test_approx_eq w.data * x.data .+ b.data y.data

                    if anygrads(wtype, xtype, btype)
                        w, x, b = wtype(randn(m, n)), xtype(randn(n, k)), btype(randn(m))
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
test_affine()
