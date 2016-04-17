using Flimsy
using Iterators

facts("plus (binary)") do
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
                            scope = DataScope()
                            gscope = GradScope()

                            a = atype <: DataVariable ? atype(randn(m, n)) : atype(randn(m, n), zeros(m, n))
                            b = btype <: DataVariable ? btype(randn(m, n)) : btype(randn(m, n), zeros(m, n))
                            c = plus(scope, a, b)
                            @fact isa(c, DataVariable) --> true
                            @fact size(c)              --> (m, n)
                            @fact c.data               --> roughly(a.data .+ b.data)

                            a = atype <: DataVariable ? atype(randn(m, n)) : atype(randn(m, n), zeros(m, n))
                            b = btype <: DataVariable ? btype(randn(m, n)) : btype(randn(m, n), zeros(m, n))
                            c = plus(gscope, a, b)
                            if anygrads(atype, btype)
                                @fact isa(c, GradVariable) --> true
                            else
                                @fact isa(c, DataVariable) --> true
                            end
                            @fact size(c) --> (m, n)
                            @fact c.data  --> roughly(a.data .+ b.data)

                            if anygrads(atype, btype)
                                context("Gradient") do
                                    a = atype <: DataVariable ? atype(randn(m, n)) : atype(randn(m, n), zeros(m, n))
                                    b = btype <: DataVariable ? btype(randn(m, n)) : btype(randn(m, n), zeros(m, n))
                                    wrt = []
                                    atype <: GradVariable && push!(wrt, a)
                                    btype <: GradVariable && push!(wrt, b)
                                    test_op_grad_mse(plus, a, b, wrt=wrt)
                                end
                            end
                        end

                        context("1xN + MxN") do
                            scope = DataScope()
                            gscope = GradScope()

                            a = atype <: DataVariable ? atype(randn(1, n)) : atype(randn(1, n), zeros(1, n))
                            b = btype <: DataVariable ? btype(randn(m, n)) : btype(randn(m, n), zeros(m, n))
                            c = plus(scope, a, b)
                            @fact isa(c, DataVariable) --> true
                            @fact size(c)              --> (m, n)
                            @fact c.data               --> roughly(a.data .+ b.data)

                            a = atype <: DataVariable ? atype(randn(1, n)) : atype(randn(1, n), zeros(1, n))
                            b = btype <: DataVariable ? btype(randn(m, n)) : btype(randn(m, n), zeros(m, n))
                            c = plus(gscope, a, b)
                            if anygrads(atype, btype)
                                @fact isa(c, GradVariable) --> true
                            else
                                @fact isa(c, DataVariable) --> true
                            end
                            @fact size(c) --> (m, n)
                            @fact c.data  --> roughly(a.data .+ b.data)

                            if anygrads(atype, btype)
                                context("Gradient") do
                                    a = atype <: DataVariable ? atype(randn(1, n)) : atype(randn(1, n), zeros(1, n))
                                    b = btype <: DataVariable ? btype(randn(m, n)) : btype(randn(m, n), zeros(m, n))
                                    wrt = []
                                    atype <: GradVariable && push!(wrt, a)
                                    btype <: GradVariable && push!(wrt, b)
                                    test_op_grad_mse(plus, a, b, wrt=wrt)
                                end
                            end
                        end

                        context("MxN + 1xN") do
                            scope = DataScope()
                            gscope = GradScope()

                            a = atype <: DataVariable ? atype(randn(m, n)) : atype(randn(m, n), zeros(m, n))
                            b = btype <: DataVariable ? btype(randn(1, n)) : btype(randn(1, n), zeros(1, n))
                            c = plus(scope, a, b)
                            @fact isa(c, DataVariable) --> true
                            @fact size(c)              --> (m, n)
                            @fact c.data               --> roughly(a.data .+ b.data)

                            a = atype <: DataVariable ? atype(randn(m, n)) : atype(randn(m, n), zeros(m, n))
                            b = btype <: DataVariable ? btype(randn(1, n)) : btype(randn(1, n), zeros(1, n))
                            c = plus(gscope, a, b)
                            if anygrads(atype, btype)
                                @fact isa(c, GradVariable) --> true
                            else
                                @fact isa(c, DataVariable) --> true
                            end
                            @fact size(c) --> (m, n)
                            @fact c.data  --> roughly(a.data .+ b.data)

                            if anygrads(atype, btype)
                                context("Gradient") do
                                    a = atype <: DataVariable ? atype(randn(m, n)) : atype(randn(m, n), zeros(m, n))
                                    b = btype <: DataVariable ? btype(randn(1, n)) : btype(randn(1, n), zeros(1, n))
                                    wrt = []
                                    atype <: GradVariable && push!(wrt, a)
                                    btype <: GradVariable && push!(wrt, b)
                                    test_op_grad_mse(plus, a, b, wrt=wrt)
                                end
                            end
                        end

                        context("Mx1 + MxN") do
                            scope = DataScope()
                            gscope = GradScope()

                            a = atype <: DataVariable ? atype(randn(m, 1)) : atype(randn(m, 1), zeros(m, 1))
                            b = btype <: DataVariable ? btype(randn(m, n)) : btype(randn(m, n), zeros(m, n))
                            c = plus(scope, a, b)
                            @fact isa(c, DataVariable) --> true
                            @fact size(c)              --> (m, n)
                            @fact c.data               --> roughly(a.data .+ b.data)

                            a = atype <: DataVariable ? atype(randn(m, 1)) : atype(randn(m, 1), zeros(m, 1))
                            b = btype <: DataVariable ? btype(randn(m, n)) : btype(randn(m, n), zeros(m, n))
                            c = plus(gscope, a, b)
                            if anygrads(atype, btype)
                                @fact isa(c, GradVariable) --> true
                            else
                                @fact isa(c, DataVariable) --> true
                            end
                            @fact size(c) --> (m, n)
                            @fact c.data  --> roughly(a.data .+ b.data)

                            if anygrads(atype, btype)
                                context("Gradient") do
                                    a = atype <: DataVariable ? atype(randn(m, 1)) : atype(randn(m, 1), zeros(m, 1))
                                    b = btype <: DataVariable ? btype(randn(m, n)) : btype(randn(m, n), zeros(m, n))
                                    wrt = []
                                    atype <: GradVariable && push!(wrt, a)
                                    btype <: GradVariable && push!(wrt, b)
                                    test_op_grad_mse(plus, a, b, wrt=wrt)
                                end
                            end
                        end

                        context("MxN + Mx1") do
                            scope = DataScope()
                            gscope = GradScope()

                            a = atype <: DataVariable ? atype(randn(m, n)) : atype(randn(m, n), zeros(m, n))
                            b = btype <: DataVariable ? btype(randn(m, 1)) : btype(randn(m, 1), zeros(m, 1))
                            c = plus(scope, a, b)
                            @fact isa(c, DataVariable) --> true
                            @fact size(c)              --> (m, n)
                            @fact c.data               --> roughly(a.data .+ b.data)

                            a = atype <: DataVariable ? atype(randn(m, n)) : atype(randn(m, n), zeros(m, n))
                            b = btype <: DataVariable ? btype(randn(m, 1)) : btype(randn(m, 1), zeros(m, 1))
                            c = plus(gscope, a, b)
                            if anygrads(atype, btype)
                                @fact isa(c, GradVariable) --> true
                            else
                                @fact isa(c, DataVariable) --> true
                            end
                            @fact size(c) --> (m, n)
                            @fact c.data  --> roughly(a.data .+ b.data)

                            if anygrads(atype, btype)
                                context("Gradient") do
                                    a = atype <: DataVariable ? atype(randn(m, n)) : atype(randn(m, n), zeros(m, n))
                                    b = btype <: DataVariable ? btype(randn(m, 1)) : btype(randn(m, 1), zeros(m, 1))
                                    wrt = []
                                    atype <: GradVariable && push!(wrt, a)
                                    btype <: GradVariable && push!(wrt, b)
                                    test_op_grad_mse(plus, a, b, wrt=wrt)
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

facts("plus (k-ary)") do
    for (K, m, n) in [(4, 3, 1), (5, 3, 7)]
        context("$(K)x$(m)x$(n)") do
            for types in map(collect, product(repeated((DataVariable, GradVariable), K)...))
                ctxstr = join(map(types) do typ
                    typ <: DataVariable ? "DataVariable" : "GradVariable"
                end, ",")
                context(ctxstr) do
                    context("MxN + ... + MxN") do
                        scope = DataScope()
                        gscope = GradScope()

                        xs = map(types) do typ
                            typ <: DataVariable ? DataVariable(randn(m, n)) : GradVariable(randn(m, n), zeros(m, n))
                        end
                        y = plus(scope, xs)
                        @fact isa(y, DataVariable) --> true
                        @fact size(y)              --> (m, n)
                        @fact y.data               --> roughly(sum(map(x->x.data, xs)))

                        xs = map(types) do typ
                            typ <: DataVariable ? DataVariable(randn(m, n)) : GradVariable(randn(m, n), zeros(m, n))
                        end
                        y = plus(gscope, xs)
                        if anygrads(types)
                            @fact isa(y, GradVariable) --> true
                        else
                            @fact isa(y, DataVariable) --> true
                        end
                        @fact size(y) --> (m, n)
                        @fact y.data  --> roughly(sum(map(x->x.data, xs)))
                        
                        if anygrads(types)
                            xs = map(types) do typ
                                typ <: DataVariable ? DataVariable(randn(m, n)) : GradVariable(randn(m, n), zeros(m, n))
                            end
                            wrt = []
                            for i = 1:K
                                types[i] <: GradVariable && push!(wrt, xs[i])
                            end
                            test_op_grad_mse(plus, xs, wrt=wrt)
                        end
                    end

                    context("MxN + Mx1 + ... + Mx1") do
                        scope = DataScope()
                        gscope = GradScope()

                        xs = map(1:K) do k
                            _m, _n = k == 1 ? (m, n) : (m, 1)
                            if types[k] <: DataVariable 
                                return DataVariable(randn(_m, _n))
                            else
                                return GradVariable(randn(_m, _n), zeros(_m, _n))
                            end
                        end
                        y = plus(scope, xs)
                        @fact isa(y, DataVariable) --> true
                        @fact size(y)              --> (m, n)
                        @fact y.data               --> roughly(plus(map(x->x.data, xs)))

                        xs = map(1:K) do k
                            _m, _n = k == 1 ? (m, n) : (m, 1)
                            if types[k] <: DataVariable 
                                return DataVariable(randn(_m, _n))
                            else
                                return GradVariable(randn(_m, _n), zeros(_m, _n))
                            end
                        end
                        y = plus(gscope, xs)
                        if anygrads(types)
                            @fact isa(y, GradVariable) --> true
                        else
                            @fact isa(y, DataVariable) --> true
                        end
                        @fact size(y) --> (m, n)
                        @fact y.data  --> roughly(plus(map(x->x.data, xs)))
                        
                        if anygrads(types)
                            xs = map(1:K) do k
                                _m, _n = k == 1 ? (m, n) : (m, 1)
                                if types[k] <: DataVariable 
                                    return DataVariable(randn(_m, _n))
                                else
                                    return GradVariable(randn(_m, _n), zeros(_m, _n))
                                end
                            end
                            wrt = []
                            for i = 1:K
                                types[i] <: GradVariable && push!(wrt, xs[i])
                            end
                            test_op_grad_mse(plus, xs, wrt=wrt)
                        end
                    end

                    context("MxN + ... + MxN + Mx1") do
                        scope = DataScope()
                        gscope = GradScope()

                        xs = map(1:K) do k
                            _m, _n = k < K ? (m, n) : (m, 1)
                            if types[k] <: DataVariable 
                                return DataVariable(randn(_m, _n))
                            else
                                return GradVariable(randn(_m, _n), zeros(_m, _n))
                            end
                        end
                        y = plus(scope, xs)
                        @fact isa(y, DataVariable) --> true
                        @fact size(y)              --> (m, n)
                        @fact y.data               --> roughly(plus(map(x->x.data, xs)))

                        xs = map(1:K) do k
                            _m, _n = k < K ? (m, n) : (m, 1)
                            if types[k] <: DataVariable 
                                return DataVariable(randn(_m, _n))
                            else
                                return GradVariable(randn(_m, _n), zeros(_m, _n))
                            end
                        end
                        y = plus(gscope, xs)
                        if anygrads(types)
                            @fact isa(y, GradVariable) --> true
                        else
                            @fact isa(y, DataVariable) --> true
                        end
                        @fact size(y) --> (m, n)
                        @fact y.data  --> roughly(plus(map(x->x.data, xs)))
                        
                        if anygrads(types)
                            xs = map(1:K) do k
                                _m, _n = k < K ? (m, n) : (m, 1)
                                if types[k] <: DataVariable 
                                    return DataVariable(randn(_m, _n))
                                else
                                    return GradVariable(randn(_m, _n), zeros(_m, _n))
                                end
                            end
                            wrt = []
                            for i = 1:K
                                types[i] <: GradVariable && push!(wrt, xs[i])
                            end
                            test_op_grad_mse(plus, xs, wrt=wrt)
                        end
                    end
                end
            end
        end
    end
end
