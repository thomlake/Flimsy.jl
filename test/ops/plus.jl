using Flimsy
using Iterators
using Base.Test

function test_binary_plus()
    for (m, n) in [(3, 1), (4, 5)]
        for atype in [DataVariable, GradVariable]
            for btype in [DataVariable, GradVariable]
                a, b = n == 1 ? (atype(randn(m)), btype(randn(m))) : (atype(randn(m, n)), btype(randn(m, n)))
                c = plus(a, b)
                @test isa(c, DataVariable)
                @test size(c) == (m, n)
                @test_approx_eq a.data .+ b.data c.data

                a, b = n == 1 ? (atype(randn(m)), btype(randn(m))) : (atype(randn(m, n)), btype(randn(m, n)))
                c = plus(CallbackStack(), a, b)
                @test isa(c, anygrads(atype, btype) ? GradVariable : DataVariable)
                @test size(c) == (m, n)
                @test_approx_eq a.data .+ b.data c.data

                if anygrads(atype, btype)
                    a, b = n == 1 ? (atype(randn(m)), btype(randn(m))) : (atype(randn(m, n)), btype(randn(m, n)))
                    wrt = []
                    atype <: GradVariable && push!(wrt, a)
                    btype <: GradVariable && push!(wrt, b)
                    test_op_grad_mse(plus, a, b, wrt=wrt)
                end
            end
        end
    end
end
test_binary_plus()

function test_kary_plus()
    for (K, m, n) in [(4, 3, 1), (5, 3, 7)]
        for types in map(collect, product(repeated((DataVariable, GradVariable), K)...))
            # MxN + ... + MxN
            xs = Variable[n == 1 ? types[k](randn(m)) : types[k](randn(m, n)) for k = 1:K]
            y = plus(xs)
            @test isa(y, DataVariable)
            @test size(y) == (m, n)
            @test_approx_eq sum(map(x->x.data, xs)) y.data

            xs = Variable[n == 1 ? types[k](randn(m)) : types[k](randn(m, n)) for k = 1:K]
            y = plus(CallbackStack(), xs)
            @test isa(y, anygrads(types) ? GradVariable : DataVariable)
            @test size(y) == (m, n)
            @test_approx_eq sum(map(x->x.data, xs)) y.data
            
            if anygrads(types)
                xs = Variable[n == 1 ? types[k](randn(m)) : types[k](randn(m, n)) for k = 1:K]
                wrt = []
                for i = 1:K
                    types[i] <: GradVariable && push!(wrt, xs[i])
                end
                test_op_grad_mse(plus, xs, wrt=wrt)
            end
            
            # MxN + Mx1 + ... Mx1
            xs = Variable[types[1](randn(m, n))]
            for k = 2:K
                push!(xs, types[k](randn(m)))
            end
            y_true = deepcopy(xs[1].data)
            for k = 2:K
                y_true = y_true .+ xs[k].data
            end 
            y = plus(xs)
            @test isa(y, DataVariable)
            @test size(y) == (m, n)
            @test_approx_eq y_true y.data

            xs = Variable[types[1](randn(m, n))]
            for k = 2:K
                push!(xs, types[k](randn(m)))
            end
            y_true = deepcopy(xs[1].data)
            for k = 2:K
                y_true = y_true .+ xs[k].data
            end 
            y = plus(CallbackStack(), xs)
            @test isa(y, anygrads(types) ? GradVariable : DataVariable)
            @test size(y) == (m, n)
            @test_approx_eq y_true y.data
            
            if anygrads(types)
                xs = Variable[types[1](randn(m, n))]
                for k = 2:K
                    push!(xs, types[k](randn(m)))
                end
                wrt = []
                for i = 1:K
                    types[i] <: GradVariable && push!(wrt, xs[i])
                end
                test_op_grad_mse(plus, xs, wrt=wrt)
            end

            # MxN + ... MxN + Mx1
            xs = Variable[types[k](randn(m, n)) for k = 1:K-1]
            push!(xs, types[K](randn(m)))
            y_true = deepcopy(xs[1].data)
            for k = 2:K
                y_true = y_true .+ xs[k].data
            end 
            y = plus(xs)
            @test isa(y, DataVariable)
            @test size(y) == (m, n)
            @test_approx_eq y_true y.data

            xs = Variable[types[k](randn(m, n)) for k = 1:K-1]
            push!(xs, types[K](randn(m)))
            y_true = deepcopy(xs[1].data)
            for k = 2:K
                y_true = y_true .+ xs[k].data
            end 
            y = plus(CallbackStack(), xs)
            @test isa(y, anygrads(types) ? GradVariable : DataVariable)
            @test size(y) == (m, n)
            @test_approx_eq y_true y.data
            
            if anygrads(types)
                xs = Variable[types[k](randn(m, n)) for k = 1:K-1]
                push!(xs, types[K](randn(m)))
                wrt = []
                for i = 1:K
                    types[i] <: GradVariable && push!(wrt, xs[i])
                end
                test_op_grad_mse(plus, xs, wrt=wrt)
            end
        end
    end
end
test_kary_plus()
