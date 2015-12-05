using Flimsy
using Flimsy: Components
using Base.Test

function check_single()
    n_classes, n_features, n_samples = 2, 20, 1
    X, Yid = rand(Flimsy.SampleData.MoG(n_classes, n_features), n_samples)
    Y = [yid == 2 for yid in Yid]
    theta = MultilabelClassifier(1, n_features)
    g() = gradient!(probs, theta, X[:,1], Y[1])
    c() = probs(theta, X[:,1], Y[1])[1]
    gradcheck(g, c, theta, verbose=false)
end

function check_batch()
    n_classes, n_features, n_samples = 2, 20, 10
    X, Yid = rand(Flimsy.SampleData.MoG(n_classes, n_features), n_samples)
    Y = [yid == 2 for yid in Yid]
    theta = MultilabelClassifier(1, n_features)
    g() = gradient!(probs, theta, X, Y)
    c() = probs(theta, X, Y)[1]
    gradcheck(g, c, theta, verbose=false)
end

function check_ml_single()
    n_classes, n_features, n_samples = 4, [5, 10, 3, 6], 1
    X1, Yid1 = rand(Flimsy.SampleData.MoG(2, n_features[1]), n_samples)
    X2, Yid2 = rand(Flimsy.SampleData.MoG(2, n_features[2]), n_samples)
    X3, Yid3 = rand(Flimsy.SampleData.MoG(2, n_features[3]), n_samples)
    X4, Yid4 = rand(Flimsy.SampleData.MoG(2, n_features[4]), n_samples)

    X = vcat(X1, X2, X3, X4)
    Y = [[yid == 2 for yid in yids] for yids in zip(Yid1, Yid2, Yid3, Yid4)]

    theta = MultilabelClassifier(n_classes, sum(n_features))
    g() = gradient!(probs, theta, X[:,1], Y[1])
    c() = probs(theta, X[:,1], Y[1])[1]
    gradcheck(g, c, theta, verbose=false)
end

function check_ml_batch()
    n_classes, n_features, n_samples = 4, [5, 2, 3, 4], 4
    X1, Yid1 = rand(Flimsy.SampleData.MoG(2, n_features[1]), n_samples)
    X2, Yid2 = rand(Flimsy.SampleData.MoG(2, n_features[2]), n_samples)
    X3, Yid3 = rand(Flimsy.SampleData.MoG(2, n_features[3]), n_samples)
    X4, Yid4 = rand(Flimsy.SampleData.MoG(2, n_features[4]), n_samples)
    
    X = vcat(X1, X2, X3, X4)
    Yid = hcat(Yid1, Yid2, Yid3, Yid4).'
    Y = Yid .== 2

    theta = MultilabelClassifier(n_classes, sum(n_features))
    g() = gradient!(probs, theta, X, Y)
    c() = probs(theta, X, Y)[1]
    gradcheck(g, c, theta, verbose=false)
end

function fit()
    n_classes, n_features = 2, [5, 3]
    n_train, n_test = 50, 50

    D = [Flimsy.SampleData.MoG(2, n_features[i]) for i = 1:n_classes]

    function create_data(n)
        X = Matrix{Float64}[]
        Y = Vector{Int}[]
        for i = 1:n_classes
            Xi, Yi = rand(D[i], n)
            push!(X, Xi)
            push!(Y, Yi)
        end
        return vcat(X...), hcat(Y...).' .== 2
    end

    X_train, Y_train = create_data(n_train)
    X_test, Y_test = create_data(n_test)

    theta = MultilabelClassifier(n_classes, sum(n_features))
    rmsprop = optimizer(RMSProp, theta, learning_rate=0.1, decay=0.5)
    nll_prev = Inf
    for i = 1:100
        nll_curr = gradient!(probs, theta, X_train, Y_train)[1]
        update!(rmsprop, theta)
        @test nll_curr <= nll_prev
        nll_prev = nll_curr
    end

    @test sum(Y_test .!= predict(theta, X_test)) / n_test <= 0.3
end

srand(12345)
check_single()
check_batch()
check_ml_single()
check_ml_batch()
fit()
