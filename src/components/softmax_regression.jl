
immutable SoftmaxRegression{V<:Variable} <: Component{V}
    w::V
    b::V
end

@component score(params::SoftmaxRegression, x) = affine(params.w, x, params.b)

# @flimsy predict(params::SoftmaxRegression, x) = Flimsy.Extras.argmax(score(params, x))

@component probs(params::SoftmaxRegression, x) = softmax(score(params, x))

@component cost(params::SoftmaxRegression, x, y) = Cost.categorical_cross_entropy_with_scores(score(params, x), y)
