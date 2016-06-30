# Flimsy.jl
# Simple Recurrent Neural Network
using Synthetic
using Flimsy
using Flimsy.Components
import Flimsy.Components: cost, predict


immutable Params{F<:Activation} <: Component
    clf::SoftmaxRegression
    rnn::SimpleRecurrent{F}
end

predict(scope::Scope, params::Params, xs::Vector) = @with scope [predict(params.clf, h) for h in unfold(params.rnn, xs)]

function cost(scope::Scope, params::Params, xs::Vector, ys::Vector)
    nll = 0.0
    @with scope begin
        hs = unfold(params.rnn, xs)
        for (h, y) in zip(hs, ys)
            nll += cost(params.clf, h, y)
        end
    end
    return nll
end

Params(n_out::Int, n_hid::Int, n_in::Int) = Params(
    clf=SoftmaxRegression(
        w=rand(Normal(0, 0.01), n_out, n_hid),
        b=zeros(n_out, 1),
    ),
    rnn=SimpleRecurrent(
        f=Tanh(),
        w=rand(Normal(0, 0.01), n_hid, n_in),
        u=orthonormal(1, n_hid, n_hid),
        b=zeros(n_hid, 1),
        h_init=zeros(n_hid, 1),
    )
)

function check()
    n_out, n_hid, n_in = 2, 10, 2
    x, y = rand(Synthetic.XORTask(20))
    params = Params(n_out, n_hid, n_in)
    println(params)
    check_gradients(cost, params, x, y)
end

function sequence_error_count(y_pred, y_true)
    count = 0
    length(y_true) == length(y_pred) || error("sequence length mismatch")
    for t = 1:length(y_true)
        if y_true[t] != y_pred[t][1]
            count += 1
        end
    end
    return count
end

function main()
    srand(123)
    max_epochs = 500
    converged = false
    n_epochs = 0
    print_freq = 10
    n_out, n_hid, n_in = 2, 7, 2
    n_train = 50
    n_test = 20
    xor_dist_train = Synthetic.XORTask(5:20)
    xor_dist_test = Synthetic.XORTask(100:200)
    train_data = rand(xor_dist_train, n_train)
    test_data = rand(xor_dist_test, n_train)
    indices = collect(1:n_train)

    dscope, gscope = DataScope(), GradScope()
    params = Params(n_out, n_hid, n_in)
    opt = optimizer(GradientDescent, params, learning_rate=0.1, clip=1.0, clipping_type=:scale)

    start_time = time()
    while n_epochs < max_epochs
        n_epochs += 1
        shuffle!(indices)
        nll = 0.0
        for i in indices
            x, y = train_data[i]
            nll += cost(gscope, params, x, y)
            backprop!(gscope)
            update!(opt)
        end
        if n_epochs % print_freq == 0
            errors = 0
            for (x, y) in train_data
                errors += sequence_error_count(predict(dscope, params, x), y)
            end
            println(n_epochs, ": nll => ", round(nll, 2), ", errors => ", errors)
            if errors < 1 
                converged = true
                break
            end
        end
    end
    stop_time = time()
    errors = 0
    for (x, y) in test_data
        errors += sequence_error_count(predict(dscope, params, x), y)
    end
    println("converged => ", converged)
    println("avg test error per sequence => ", errors / n_test)
    println("ran ", n_epochs, " epochs in ", round(stop_time - start_time, 2), " seconds")
end

("-c" in ARGS || "--check" in ARGS) && check()
main()
