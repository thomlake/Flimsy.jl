
"""Abstract Linear Model."""
abstract LinearModel{V} <: Component{V}

function call{C<:LinearModel}(::Type{C}, n_output::Int, n_input::Int)
    return C(w=rand(Normal(0, 0.01), n_output, n_input), b=zeros(n_output, 1))
end

@component score(params::LinearModel, x::Variable) = affine(params.w, x, params.b)

"""
Linear Regression Component.

    y|x ~ Normal(w*x + b, 1)
"""
immutable LinearRegression{V<:Variable} <: LinearModel{V}
    w::V
    b::V
    function LinearRegression(w::V, b::V)
        m, n = size(w)
        size(b) == (m, 1) || throw(DimensionMismatch("Bad size(b) == $(size(b)) != ($m, 1)"))
        return new(w, b)
    end
end
LinearRegression{V<:Variable}(w::V, b::V) = LinearRegression{V}(w, b)

@component predict(params::LinearRegression, x::Variable) = score(params, x)
@component cost(params::LinearRegression, x::Variable, y) = Cost.mse(score(params, x), y)
@component cost(params::LinearRegression, x::Variable, y, weight::Real) = Cost.mse(score(params, x), y, weight)

"""
Softmax Regression Component.
    
    y|x ~ Categorical(softmax(w*x + b))
"""
immutable SoftmaxRegression{V<:Variable} <: LinearModel{V}
    w::V
    b::V
    function SoftmaxRegression(w::V, b::V)
        m, n = size(w)
        size(b) == (m, 1) || throw(DimensionMismatch("Bad size(b) == $(size(b)) != ($m, 1)"))
        return new(w, b)
    end
end
SoftmaxRegression{V<:Variable}(w::V, b::V) = SoftmaxRegression{V}(w, b)

@component predict(params::SoftmaxRegression, x::Variable) = argmax(score(params, x))
@component probs(params::SoftmaxRegression, x::Variable) = softmax(score(params, x))
@component cost(params::SoftmaxRegression, x::Variable, y) = Cost.categorical_cross_entropy_with_scores(score(params, x), y)
@component cost(params::SoftmaxRegression, x::Variable, y, weight::Real) = Cost.categorical_cross_entropy_with_scores(score(params, x), y, weight)


"""
Sigmoid Regression Component.
    
    y_i|x ~ Bernoulli(sigmoid(w*x + b)_i)
"""
immutable SigmoidRegression{V<:Variable} <: LinearModel{V}
    w::V
    b::V
    function SigmoidRegression(w::V, b::V)
        m, n = size(w)
        size(b) == (m, 1) || throw(DimensionMismatch("Bad size(b) == $(size(b)) != ($m, 1)"))
        return new(w, b)
    end
end
SigmoidRegression{V<:Variable}(w::V, b::V) = SigmoidRegression{V}(w, b)

@component function predict(params::SigmoidRegression, x::Variable, threshold::Real=0.5)
    @blacklist log
    return score(params, x).data .> log(threshold / (1 - threshold))
end
@component probs(params::SigmoidRegression, x::Variable) = sigmoid(score(params, x))
@component cost(params::SigmoidRegression, x::Variable, y) = Cost.bernoulli_cross_entropy_with_scores(score(params, x), y)
@component cost(params::SigmoidRegression, x::Variable, y, weight::Real) = Cost.bernoulli_cross_entropy_with_scores(score(params, x), y, weight)

