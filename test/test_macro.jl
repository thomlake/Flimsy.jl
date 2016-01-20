using Flimsy

println(macroexpand(
    quote
        @Flimsy.component f(x, y, z) = foo(x + bar(y) + z)
    end
))

println(macroexpand(
    quote
        @Flimsy.component function f(θ, x)
            h = Array(eltype(x), length(x))
            h[1] = step(θ, x[1])
            for t = 2:length(x)
                h[t] = step(θ, x[t], h[t-1])
            end
            return h
        end
    end
))
