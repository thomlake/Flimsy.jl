using Flimsy
using Flimsy.Components
import Flimsy.Components: score, predict, probs

immutable SeqTagger{T,NOut,NHid} <: Component
    out::LogisticRegression{T,NOut,NHid}
    rnn::Component
end

@flimsy score(theta::SeqTagger, xs::Vector) = [score(theta.out, h) for h in unfold(theta.rnn, xs)]

@flimsy predict(theta::SeqTagger, xs::Vector) = [predict(theta.out, h)[1] for h in unfold(theta.rnn, xs)]

@flimsy probs(theta::SeqTagger, xs::Vector) = [probs(theta.out, h) for h in unfold(theta.rnn, xs)]

@flimsy function probs(theta::SeqTagger, xs::Vector, ys::Vector)
    nll = 0.0
    for (h, y) in zip(unfold(theta.rnn, xs), ys)
        nll += probs(theta.out, h, y)[1]
    end
    return nll
end

GRUTagger(n_out::Int, n_hid::Int, n_in::Int) = SeqTagger(
    LogisticRegression(n_out, n_hid),
    GatedRecurrent(n_hid, n_in),
)

LSTMTagger(n_out::Int, n_hid::Int, n_in::Int) = SeqTagger(
    LogisticRegression(n_out, n_hid),
    LSTM(n_hid, n_in),
)


function check()
    n_in = 2
    n_hid = 5
    n_out = 2
    xs, ys = rand(Flimsy.SampleData.XOr(20))

    gru = GRUTagger(n_out, n_hid, n_in)
    g() = gradient!(probs, gru, xs, ys)
    c() = probs(gru, xs, ys)[1]
    gradcheck(g, c, gru)

    lstm = LSTMTagger(n_out, n_hid, n_in)
    g() = gradient!(probs, lstm, xs, ys)
    c() = probs(lstm, xs, ys)[1]
    gradcheck(g, c, lstm)
end

function fit()
    n_in = 2
    n_hid = 5
    n_out = 2
    n_train, n_valid = 100, 20
    xor = Flimsy.SampleData.XOr(5:20)
    X_train, Y_train = rand(xor, n_train)
    X_valid, Y_valid = rand(xor, n_valid)
    minlen = minimum(map(length, X_train))
    maxlen = maximum(map(length, X_train))

    models = Dict(
        :gru => GRUTagger(n_out, n_hid, n_in),
        :lstm => LSTMTagger(n_out, n_hid, n_in),
    )
    info = Dict()

    for (name, theta) in models
        opt = optimizer(RMSProp, theta, decay=0.8, clip=3.0, clipping_type=:clip)
        progress = Flimsy.Extras.Progress(theta, patience=3) do
            errors = 0
            for (xs, ys) in zip(X_valid, Y_valid)
                errors += sum(ys .!= predict(theta, xs))
            end
            return errors
        end
        function statusmsg()
            @printf("[%s] epoch: %03d, best: %0.02f, current: %0.02f\n", name, progress.epoch, progress.best_value, progress.current_value)
        end

        indices = collect(1:n_train)
        start(progress)
        for i = 1:10
            shuffle!(indices)
            for i in indices
                xs, ys = X_train[i], Y_train[i]
                opt.learning_rate = 0.1 / length(xs)
                gradient!(probs, theta, xs, ys)
                update!(opt, theta)
            end
            step(progress, true)
            statusmsg()
        end
        done(progress)
        info[name] = progress
    end

    println("[train]")
    println("  number samples => ", n_train)
    println("  min seq length => ", minlen)
    println("  max seq length => ", maxlen)
    @printf("  cpu time     [gru => %0.02f, lstm => %0.02f, gru/lstm => %0.02f]\n", time(info[:gru]), time(info[:lstm]), time(info[:gru]) / time(info[:lstm]))
    @printf("  total errors [gru => %0.02f, lstm => %0.02f, gru/lstm => %0.02f]\n", info[:gru].best_value, info[:lstm].best_value, info[:gru].best_value / info[:lstm].best_value)
    println()

    n_test = 20
    X_test, Y_test = rand(Flimsy.SampleData.XOr(100:200), n_test)
    total_timesteps = mapreduce(length, +, Y_test)
    minlen = minimum(map(length, X_test))
    maxlen = maximum(map(length, X_test))
    errors_gru, errors_lstm = 0, 0
    for (xs, ys) in zip(X_test, Y_test)
        errors_gru += sum(ys .!= predict(info[:gru].best_model, xs))
        errors_lstm += sum(ys .!= predict(info[:lstm].best_model, xs))
    end
    println("[test]")
    println("  number samples => ", n_test)
    println("  min seq length => ", minlen)
    println("  max seq length => ", maxlen)
    @printf("  performance ratio => %0.02f\n", errors_gru / errors_lstm)
    @printf("  tot error [gru => %0.02f, lstm => %0.02f]\n", errors_gru, errors_lstm)
    @printf("  seq error [gru => %0.02f, lstm => %0.02f]\n", errors_gru / n_test, errors_lstm / n_test)
    @printf("  avg error [gru => %0.02f, lstm => %0.02f]\n", errors_gru / total_timesteps, errors_lstm / total_timesteps)
end

check()
fit()
