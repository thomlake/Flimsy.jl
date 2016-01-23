
abstract LinearModel{V} <: Component{V}

@component score(params::LinearModel, x) = affine(params.w, x, params.b)


immutable LinearRegression{V<:Variable} <: LinearModel{V}
    w::V
    b::V
end

@component predict(params::LinearRegression, x) = score(params, x)

@component cost(params::LinearRegression, x, y) = Cost.categorical_cross_entropy_with_scores(score(params, x), y)


immutable LogisticRegression{V<:Variable} <: LinearModel{V}
    w::V
    b::V
end

@component predict(params::LogisticRegression, x) = argmax(score(params, x))

@component probs(params::LogisticRegression, x) = softmax(score(params, x))

@component cost(params::LogisticRegression, x, y) = Cost.categorical_cross_entropy_with_scores(score(params, x), y)


