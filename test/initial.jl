using Flimsy

immutable Tanh{V<:Variable} <: Component
    x::V
end

@component predict(theta::Tanh) = tanh(theta.x)

@component cost(theta::Tanh, y) = Cost.mse(tanh(theta.x), y)

function main_tanh()
    n = 10
    x = GradInput(reshape(linspace(-1, 1, n), n))
    y = randn(n)
    g() = gradient!(Cost.mse, x, y)
    c() = Cost.mse(x, y)
    check_gradients(g, c, x)
end
main_tanh()

# function main_sum()
#     scope = Flimsy.Scope(Flimsy.GradValue)
#     a1 = reshape(linspace(-1, 1, 10), 5, 2)
#     b1 = reshape(linspace(-1, 1, 10), 5, 2)
#     A1 = Flimsy.InputVariable(scope, a1)
#     B1 = Flimsy.SharedVariable(scope, b1)
#     C1 = sum(scope, A1, B1)
#     println(C1)
#     c1 = Flimsy.lookup(scope, C1)
#     println(all(c1.data .== a1 .+ b1))

#     a2 = reshape(linspace(-1, 1, 10), 5, 2)
#     b2 = reshape(linspace(-1, 1, 10), 5, 2)
#     A2 = Flimsy.InputVariable(scope, a2)
#     B2 = Flimsy.SharedVariable(scope, b2)
#     C2 = sum(scope, A2, B2)
#     println(C2)
#     c2 = Flimsy.lookup(scope, C2)
#     println(all(c2.data .== a2 .+ b2))

#     target = randn(5, 2)
#     nll = Flimsy.mse(scope, B2, target)
#     println(nll, ", ", sum(abs2(Flimsy.lookup(scope, B2).data .- target)))

#     nll = Flimsy.mse(scope, C2, target)
#     println(nll, ", ", sum(abs2(c2.data .- target)))

#     Flimsy.backprop!(scope)

#     @code_warntype sum(scope, A1, B1)
# end

# function main_stupid()
#     n, m, b = 50, 200, 20
#     dataScope = Flimsy.Scope()
#     W = [Flimsy.SharedVariable(dataScope, randn(m, b)) for i = 1:n]

#     scope = Flimsy.GradScope(dataScope)
#     epoch = 0
#     while true
#         X = Flimsy.InputVariable(scope, randn(m, b))
#         y = target = randn(m, b)
#         for i = 1:n
#             X = sum(scope, W[i], X)
#         end
#         nll = Flimsy.mse(scope, X, y)
#         Flimsy.backprop!(scope)
#         Flimsy.reset!(scope)
#         epoch += 1
#         if epoch % 1000 == 0 
#             println("[$epoch] $nll, $(length(scope.inputs))")
#             # whos()
#         end
#     end
# end
