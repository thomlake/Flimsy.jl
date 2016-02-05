using Flimsy

# model definition
immutable Params{V<:Variable} <: Component{V}
    w::V
    b::V
end

# default constructor
Params(m, n) = Params(w=randn(m, n), b=zeros(m, 1))

# computation the model performs
@component score(θ::Params, x::Variable) = affine(θ.w, x, θ.b)
@component predict(θ::Params, x::Variable) = argmax(score(θ, x))
@component probs(θ::Params, x::Variable) = softmax(score(θ, x))
@component cost(θ::Params, x::Variable, y) = Cost.categorical_cross_entropy_with_scores(score(θ, x), y)

# check gradients using finite differences
function check()
    n_samples, n_classes, n_features = 5, 3, 20
    X, Y = rand(Flimsy.Demo.MoG(n_classes, n_features), n_samples)
    θ = Params(n_classes, n_features)
    scope = Scope()
    g = () -> gradient!(cost, scope, θ, Input(X), Y)
    c = () -> cost(reset!(scope), θ, Input(X), Y)
    check_gradients(g, c, θ)
end

# train and test
function main()
    srand(sum(map(Int, collect("Flimsy"))))
    n_classes, n_features = 3, 20
    n_train, n_test = 50, 50
    D = Flimsy.Demo.MoG(n_classes, n_features)
    X_train, Y_train = rand(D, n_train)
    X_test, Y_test = rand(D, n_test)

    θ = Params(n_classes, n_features)
    scope = Scope()
    opt = optimizer(RMSProp, θ, learning_rate=0.01, decay=0.9)
    for i = 1:100
        nll = gradient!(cost, scope, θ, Input(X_train), Y_train)
        update!(opt)
        i % 10 == 0 && println("epoch => $i, nll => $nll")
    end
    println("train error => ", sum(Y_train .!= predict(reset!(scope), θ, Input(X_train))) / n_train)
    println("test error  => ", sum(Y_test .!= predict(reset!(scope), θ, Input(X_test))) / n_test)
end

("-c" in ARGS || "--check" in ARGS) && check()
main()