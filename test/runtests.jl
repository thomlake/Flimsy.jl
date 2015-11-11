using Flimsy
using Base.Test

tests = [
    "var",
    "getparams",
    "ops/identity",
    "ops/tanh",
    "ops/sigmoid",
    "ops/relu",
    "ops/softmax",
    "ops/wta",
    "ops/linear",
    "ops/prod",
    "ops/sum",
    "ops/minus",
    "ops/concat",
    "ops/affine",
    "ops/dropout",
    "logisticregression",
    "ctc",
]

srand(sum(map(Int, collect("NeuralNet"))))

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

const eps = 1e-6
const tol = 1e-6

function test_op_grad(f1::Function, f2::Function, x::Flimsy.Var)
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

for t in tests
    global failed
    failed = false
    f = "$t.jl"
    print("* test $f...")
    Test.with_handler(test_handler) do
        include(f)
    end
    if !failed
        print_with_color(:green, " ok\n")
    end
end

# for t in readdir("ops/")
#     global failed
#     failed = false
#     f = joinpath("ops", t)
#     print("* test $f...")
#     Test.with_handler(test_handler) do
#         include(f)
#     end
#     if !failed
#         print_with_color(:green, " ok\n")
#     end
# end
