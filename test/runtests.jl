using Flimsy
using Base.Test

tests = [
    # "var.jl",
    # "getparams.jl",
    # "ops/identity.jl",
    "ops/tanh.jl",
    # "ops/sigmoid.jl",
    # "ops/relu.jl",
    # "ops/softmax.jl",
    # "ops/softmax_vector.jl",
    # "ops/wta.jl",
    # "ops/linear.jl",
    # "ops/affine.jl",
    # "ops/plus.jl",
    # "ops/minus_scalar.jl",
    # "ops/minus.jl",
    # "ops/mult_scalar.jl",
    # "ops/mult.jl",
    # "ops/concat.jl",
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

global failed = false

test_handler(r::Test.Success) = nothing

function test_handler(r::Test.Failure)
    global failed
    if !failed
        println()
        failed = true
    end
    print_with_color(:red, "failure: ")
    println("expr: $(r.expr), result: $(r.resultexpr)")
end

function test_handler(r::Test.Error)
    global failed
    if !failed
        println()
        failed = true
    end
    print_with_color(:red, "error:")
    showerror(STDOUT, r)
    println()
end

function test_op_grad_mse(f::Function, args...; wrt=nothing, eps=1e-6, tol=1e-6)
    if wrt == nothing
        error("wrt must be given")
    end
    if !isa(wrt, AbstractArray)
        wrt = typeof(wrt)[wrt]
    end

    scope = Flimsy.Scope(Flimsy.Components.EmptyComponent())
    output = f(scope, args...)
    target = randn(size(output))
    Cost.mse(scope.stack, output, target)

    backprop!(scope.stack)
    for x in wrt
        for i in eachindex(x)
            xi = x.data[i]
            x.data[i] = xi + eps
            lp = Cost.mse(f(args...), target)
            x.data[i] = xi - eps
            lm = Cost.mse(f(args...), target)
            x.data[i] = xi
            dx = (lp - lm) / (2 * eps)
            @test abs(dx - x.grad[i]) < tol
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
    global failed
    failed = false
    print("* test $tf...")
    if isfile(tf)
        Test.with_handler(test_handler) do
            include(tf)
        end
        if !failed
            print_with_color(:green, " ok\n")
        end
    else
        print_with_color(:blue, " no file\n")
    end
end
