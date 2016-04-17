include("../src/Flimsy.jl")
using Flimsy
using FactCheck

FactCheck.setstyle(:compact)

tests = [
    "inplace.jl",
    "ops/identity.jl",
    "ops/tanh.jl",
    "ops/sigmoid.jl",
    "ops/relu.jl",
    "ops/softmax.jl",
    "ops/softmax_vector.jl",
    "ops/wta.jl",
    "ops/recip.jl",
    "ops/norm2.jl",
    "ops/exp.jl",
    "ops/log.jl",
    "ops/linear.jl",
    "ops/affine.jl",
    "ops/plus.jl",
    "ops/minus_scalar.jl",
    "ops/minus.jl",
    "ops/mult_scalar.jl",
    "ops/mult.jl",
    "ops/concat.jl",
    "ops/decat.jl",
    "ops/embed.jl",
    "ops/dot.jl",
    # "ops/dropout.jl",
    "mse.jl",
    "categorical_cross_entropy.jl",
    "categorical_cross_entropy_with_scores.jl",
    "ctc.jl",
]

srand(sum(map(Int, collect("Flimsy"))))

function test_op_grad_mse(f::Function, args...; wrt=nothing, eps=1e-3, atol=0.1, rtol=0.1)
    if wrt == nothing
        error("wrt must be given")
    end

    if !isa(wrt, AbstractArray)
        wrt = typeof(wrt)[wrt]
    end

    scope = DataScope()
    gradscope = GradScope()
    output = f(gradscope, args...)
    target = map(FloatX, randn(size(output)))
    Cost.mse(gradscope, output, target)
    backprop!(gradscope)

    eps = FloatX(eps)
    atol = FloatX(atol)
    rtol = FloatX(rtol)

    for x in wrt
        for i in eachindex(x)
            xi = x.data[i]
            x.data[i] = xi + eps
            lp = Cost.mse(scope, f(scope, args...), target)
            x.data[i] = xi - eps
            lm = Cost.mse(scope, f(scope, args...), target)
            x.data[i] = xi
            dx = FloatX(lp - lm) / FloatX(2 * eps)
            if abs(dx) > 1
                ratio = dx / x.grad[i]
                @fact ratio --> roughly(1, rtol) "$dx != $(x.grad[i]) at index $i"
            else
                @fact dx --> roughly(x.grad[i], atol) "$dx != $(x.grad[i]) at index $i"
            end
        end
    end
end

function runtests()
    for tf in tests
        if isfile(tf)
            include(tf)
        else
            print_with_color(:blue, "* test $tf... no file\n")
        end
    end
end
runtests()
