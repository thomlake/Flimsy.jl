using Flimsy
using Base.Test

tests = [
    "var.jl",
    "getparams.jl",
    "ops/identity.jl",
    "ops/tanh.jl",
    "ops/sigmoid.jl",
    "ops/relu.jl",
    "ops/softmax.jl",
    "ops/wta.jl",
    "ops/linear.jl",
    "ops/prod.jl",
    "ops/sum.jl",
    "ops/minus.jl",
    "ops/concat.jl",
    "ops/decat.jl",
    "ops/affine.jl",
    "ops/dropout.jl",
    "ndembed.jl",
    "logistic_regression.jl",
    "multilabel_classifier.jl",
    "ctc.jl",
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


function test_op_grad(f1::Function, f2::Function, x::Variable; eps::AbstractFloat=1e-6, tol::AbstractFloat=1e-6)
    stack = BPStack()
    y = f1(stack)
    t = randn(size(y))
    Flimsy.Cost.gauss(BPStack(), t, y)
    backprop!(stack)
    for i in eachindex(x)
        xi = x.data[i]
        x.data[i] = xi + eps
        lp = Flimsy.Cost.gauss(t, f2())
        x.data[i] = xi - eps
        lm = Flimsy.Cost.gauss(t, f2())
        x.data[i] = xi
        dx = (lp - lm) / (2 * eps)
        @test abs(dx - x.grad[i]) < tol
    end
end

function test_op_grad(f1::Function, f2::Function, x::Vector{Variable}; eps::AbstractFloat=1e-6, tol::AbstractFloat=1e-6)
    stack = BPStack()
    y = f1(stack)
    n = length(y)
    t = [randn(size(y[i])) for i = 1:n]
    for i = 1:n Flimsy.Cost.gauss(BPStack(), t[i], y[i]) end
    backprop!(stack)
    for i = 1:n
        for j in eachindex(x[i]) 
            x_ij = x[i].data[j]
            x[i].data[j] = x_ij + eps
            yp = f2()
            lp = sum([Flimsy.Cost.gauss(t[k], yp[k]) for k = 1:n])
            x[i].data[j] = x_ij - eps
            ym = f2()
            lm = sum([Flimsy.Cost.gauss(t[k], ym[k]) for k = 1:n])
            x[i].data[j] = x_ij
            dx_ij = (lp - lm) / (2 * eps)
            @test abs(dx_ij - x[i].grad[j]) < tol
        end
    end
end

for tf in tests
    global failed
    failed = false
    print("* test $tf...")
    Test.with_handler(test_handler) do
        include(tf)
    end
    if !failed
        print_with_color(:green, " ok\n")
    end
end
