using Flimsy
using Synthetic

# Parameter definition
immutable Params{V<:Variable} <: Component{V}
    w::V
    b::V
end

# Default constructor
Params(m, n) = Params(w=randn(m, n), b=zeros(m, 1))

# Computation the model performs
@component score(θ::Params, x::Variable) = affine(θ.w, x, θ.b)
@component predict(θ::Params, x::Variable) = argmax(score(θ, x))
@component probs(θ::Params, x::Variable) = softmax(score(θ, x))
@component cost(θ::Params, x::Variable, y) = Cost.categorical_cross_entropy_with_scores(score(θ, x), y)

# Check gradients using finite differences
function check()
    println("checking gradients....")
    n_features, n_classes, n_samples = 20, 3, 5
    data = rand(Synthetic.MixtureTask(n_features, n_classes), n_samples)
    X = hcat(map(first, data)...)
    y = vcat(map(last, data)...)
    θ = setup(Params(n_classes, n_features))
    check_gradients(cost, θ, Input(X), y)
end

# Train/Test
function main()
    srand(sum(map(Int, collect("Flimsy"))))
    n_features, n_classes = 20, 3
    n_train, n_test = 50, 50
    D = Synthetic.MixtureTask(n_features, n_classes)
    data_train = rand(D, n_train)
    data_test = rand(D, n_test)
    X_train, y_train = hcat(map(first, data_train)...), vcat(map(last, data_train)...)
    X_test, y_test = hcat(map(first, data_test)...), vcat(map(last, data_test)...)

    θ = setup(Params(n_classes, n_features))
    opt = optimizer(RmsProp, θ, learning_rate=0.01, decay=0.9)
    start_time = time()
    for i = 1:100
        nll = cost(θ, Input(X_train), y_train; grad=true)
        update!(opt)
        i % 10 == 0 && println("epoch => $i, nll => $nll")
    end
    println("wall time   => ", time() - start_time)
    println("train error => ", sum(y_train .!= predict(θ, Input(X_train))) / n_train)
    println("test error  => ", sum(y_test .!= predict(θ, Input(X_test))) / n_test)
end

("-c" in ARGS || "--check" in ARGS) && check()
main()
