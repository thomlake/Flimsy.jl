# Flimsy.jl
# Simple Recurrent Neural Network

using Flimsy
using Flimsy.Components
import Flimsy.Components: cost, predict
import Flimsy.Demo: XORTask

immutable Params{T} <: Component{T}
    clf::LogisticRegression{T}
    rnn::SimpleRecurrent{T}
end

@component predict(params::Params, xs::Vector) = [predict(params.clf, h) for h in unfold(params.rnn, xs)]

@component function cost{I<:Integer}(params::Params, xs::Vector, ys::Vector{I})
    nll = 0.0
    hs = unfold(params.rnn, xs)
    for (h, y) in zip(hs, ys)
        nll += cost(params.clf, h, y)
    end
    return nll
end

Params(n_out::Int, n_hid::Int, n_in::Int) = Params(
    clf=LogisticRegression(
        w=rand(Normal(0, 0.01), n_out, n_hid),
        b=zeros(n_out),
    ),
    rnn=SimpleRecurrent(
        f=tanh,
        # w=rand(Normal(0, 0.1), n_hid, n_in),
        # u=rand(Normal(0, 0.1), n_hid, n_hid),
        w=orthonormal(1, n_hid, n_in),
        u=orthonormal(1, n_hid, n_hid),
        b=zeros(n_hid),
        h0=zeros(n_hid),
    )
)

function check()
    n_out, n_hid, n_in = 2, 10, 2
    xs, ys = rand(XORTask(20))
    params = Params(n_out, n_hid, n_in)
    println("Params:")
    for (name, param) in getnamedparams(params)
        println("  $name => ", size(param))
    end
    g = () -> gradient!(cost, params, map(Input, xs), ys)
    c = () -> cost(params, map(Input, xs), ys)
    check_gradients(g, c, params)
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

function fit()
    srand(1235)
    n_out, n_hid, n_in = 2, 7, 2
    n_train = 20
    xortask = XORTask(20)
    D = collect(zip(rand(xortask, n_train)...))

    params = Params(n_out, n_hid, n_in)
    opt = optimizer(GradientDescent, params, learning_rate=0.05, clip=5.0, clipping_type=:scale)
    progress
    indices = collect(1:n_train)
    tic()
    for epoch = 1:500
        shuffle!(indices)
        nll = 0.0
        for i in indices
            x, y = D[i]
            nll += gradient!(cost, params, map(Input, x), y)
            update!(opt, params)
        end
        epoch += 1
        if epoch % 10 == 0
            errors = 0
            for (x, y) in D
                errors += sequence_error_count(predict(params, map(Input, x)), y)
            end
            println("[$epoch] errors: $errors, cost: $nll")
            errors == 0 && break
        end
    end
    toc()
end

check()
fit()
