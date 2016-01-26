
abstract LinearModel{V} <: Component{V}
@component score(params::LinearModel, x::Variable) = affine(params.w, x, params.b)

immutable LinearRegression{V<:Variable} <: LinearModel{V}
    w::V
    b::V
end
@component predict(params::LinearRegression, x::Variable) = score(params, x)
@component cost(params::LinearRegression, x::Variable, y) = Cost.mse(score(params, x), y)
@component cost(params::LinearRegression, x::Variable, y, weight::Real) = Cost.mse(score(params, x), y, weight)

immutable SoftmaxRegression{V<:Variable} <: LinearModel{V}
    w::V
    b::V
end
@component predict(params::SoftmaxRegression, x::Variable) = argmax(score(params, x))
@component probs(params::SoftmaxRegression, x::Variable) = softmax(score(params, x))
@component cost(params::SoftmaxRegression, x::Variable, y) = Cost.categorical_cross_entropy_with_scores(score(params, x), y)
@component cost(params::SoftmaxRegression, x::Variable, y, weight::Real) = Cost.categorical_cross_entropy_with_scores(score(params, x), y, weight)

immutable SigmoidRegression{V<:Variable} <: LinearModel{V}
    w::V
    b::V
end
@component predict(params::SigmoidRegression, x::Variable, threshold::Real=0.5) = score(params, x).data .> log(threshold / (1 - threshold))
@component probs(params::SigmoidRegression, x::Variable) = sigmoid(score(params, x))
@component cost(params::SigmoidRegression, x::Variable, y) = Cost.bernoulli_cross_entropy_with_scores(score(params, x), y)
@component cost(params::SigmoidRegression, x::Variable, y, weight::Real) = Cost.bernoulli_cross_entropy_with_scores(score(params, x), y, weight)


