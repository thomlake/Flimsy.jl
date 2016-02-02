using Flimsy
using FactCheck

# FactCheck.setstyle(:compact)

tests = [
    # "var.jl",
    # "getparams.jl",
    # "ops/identity.jl",
    "ops/tanh.jl",
    # "ops/sigmoid.jl",
    # "ops/relu.jl",
    "ops/softmax.jl",
    "ops/softmax_vector.jl",
    # "ops/wta.jl",
    "ops/linear.jl",
    # "ops/affine.jl",
    # "ops/plus.jl",
    # "ops/minus_scalar.jl",
    # "ops/minus.jl",
    # "ops/mult_scalar.jl",
    # "ops/mult.jl",
    "ops/concat.jl",
    # "ops/decat.jl",
    # "ops/dropout.jl",
    # "mse.jl",
    # "categorical_cross_entropy.jl",
    # "categorical_cross_entropy_with_scores.jl",
    # "ndembed.jl",
    # "logistic_regression.jl",
    # "multilabel_classifier.jl",
    # "ctc.jl",
]

srand(sum(map(Int, collect("Flimsy"))))

function test_op_grad_mse(f::Function, args...; wrt=nothing, eps=1e-6, tol=1e-6)
    if wrt == nothing
        error("wrt must be given")
    end
    if !isa(wrt, AbstractArray)
        wrt = typeof(wrt)[wrt]
    end

    scope = DynamicScope()
    gradscope = GradScope(scope)
    output = f(gradscope, args...)
    target = randn(size(output))
    Cost.mse(gradscope, output, target)
    backprop!(gradscope)

    for x in wrt
        for i in eachindex(x)
            xi = x.data[i]
            x.data[i] = xi + eps
            lp = Cost.mse(scope, f(scope, args...), target)
            x.data[i] = xi - eps
            lm = Cost.mse(scope, f(scope, args...), target)
            x.data[i] = xi
            dx = (lp - lm) / (2 * eps)
            @fact abs(dx - x.grad[i]) --> less_than(tol)
        end
    end


    # stack = CallbackStack()
    # output = f(stack, args...)
    # target = randn(size(output))
    # Cost.mse(stack, output, target)
    # backprop!(stack)
    # for x in wrt
    #     for i in eachindex(x)
    #         xi = x.data[i]
    #         x.data[i] = xi + eps
    #         lp = Cost.mse(f(args...), target)
    #         x.data[i] = xi - eps
    #         lm = Cost.mse(f(args...), target)
    #         x.data[i] = xi
    #         dx = (lp - lm) / (2 * eps)
    #         @test abs(dx - x.grad[i]) < tol
    #     end
    # end
end

for tf in tests
    if isfile(tf)
        include(tf)
    else
        print_with_color(:blue, "* test $tf... no file\n")
    end
end
