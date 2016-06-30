
"""Abstract Linear Model."""
abstract LinearModel <: Component

function call{C<:LinearModel}(::Type{C}, n_output::Int, n_input::Int)
    return C(w=rand(Normal(0, 0.01), n_output, n_input), b=zeros(n_output, 1))
end

score(scope::Scope, params::LinearModel, x::Variable) = @with scope affine(params.w, x, params.b)
score(scope::Scope, params::LinearModel, x::Variable, p) = @with scope affine(params.w, dropout(x, p), params.b)

"""
Linear Regression Component.

    y|x ~ Normal(f(x), σ)
    f(x) = W * x + b
"""
immutable LinearRegression <: LinearModel
    w::GradVariable
    b::GradVariable
    function LinearRegression(w::GradVariable, b::GradVariable)
        m, n = size(w)
        size(b) == (m, 1) || throw(DimensionMismatch("Bad size(b) == $(size(b)) != ($m, 1)"))
        return new(w, b)
    end
end

predict(scope::Scope, params::LinearRegression, x::Variable) = @with scope score(params, x)
predict(scope::Scope, params::LinearRegression, x::Variable, p) = @with scope score(params, x, p)

cost(scope::Scope, params::LinearRegression, x::Variable, y) = @with scope Cost.mse(score(params, x), y)
cost(scope::Scope, params::LinearRegression, x::Variable, y, p) = @with scope Cost.mse(score(params, x, p), y)

weighted_cost(scope::Scope, params::LinearRegression, x::Variable, y, weight::AbstractFloat) = @with scope Cost.mse(score(params, x), y, weight)
weighted_cost(scope::Scope, params::LinearRegression, x::Variable, y, p, weight::AbstractFloat) = @with scope Cost.mse(score(params, x, p), y, weight)

"""
Softmax Regression Component.
    
    y|x ~ Categorical(f(x))
    f(x) = softmax(w * x + b)
"""
immutable SoftmaxRegression <: LinearModel
    w::GradVariable
    b::GradVariable
    function SoftmaxRegression(w::GradVariable, b::GradVariable)
        m, n = size(w)
        size(b) == (m, 1) || throw(DimensionMismatch("Bad size(b) == $(size(b)) != ($m, 1)"))
        return new(w, b)
    end
end

predict(scope::Scope, params::SoftmaxRegression, x::Variable) = @with scope argmax(score(params, x))
predict(scope::Scope, params::SoftmaxRegression, x::Variable, p) = @with scope argmax(score(params, x, p))

probs(scope::Scope, params::SoftmaxRegression, x::Variable) = @with scope softmax(score(params, x))
probs(scope::Scope, params::SoftmaxRegression, x::Variable, p) = @with scope softmax(score(params, x, p))

cost(scope::Scope, params::SoftmaxRegression, x::Variable, y) = @with scope Cost.categorical_cross_entropy_with_scores(score(params, x), y)
cost(scope::Scope, params::SoftmaxRegression, x::Variable, y, p) = @with scope Cost.categorical_cross_entropy_with_scores(score(params, x, p), y)

weighted_cost(scope::Scope, params::SoftmaxRegression, x::Variable, y, weight::Real) = @with scope Cost.categorical_cross_entropy_with_scores(score(params, x), y, weight)
weighted_cost(scope::Scope, params::SoftmaxRegression, x::Variable, y, p, weight::AbstractFloat) = @with scope Cost.categorical_cross_entropy_with_scores(score(params, x, p), y, weight)

"""
Sigmoid Regression Component.
    
    y[i]|x ~ Bernoulli(f(x)[i])
    f(x) = sigmoid(w * x + b)
"""
immutable SigmoidRegression <: LinearModel
    w::GradVariable
    b::GradVariable
    function SigmoidRegression(w::GradVariable, b::GradVariable)
        m, n = size(w)
        size(b) == (m, 1) || throw(DimensionMismatch("Bad size(b) == $(size(b)) != ($m, 1)"))
        return new(w, b)
    end
end

function predict(scope::Scope, params::SigmoidRegression, x::Variable, threshold::Real=0.5)
    scores = @with scope score(params, x)
    return scores.data .> log(threshold / (1 - threshold))
end
function predict(scope::Scope, params::SigmoidRegression, x::Variable, p, threshold::Real=0.5)
    scores = @with scope score(params, x, p)
    return scores.data .> log(threshold / (1 - threshold))
end

probs(scope::Scope, params::SigmoidRegression, x::Variable) = @with scope sigmoid(score(params, x))
probs(scope::Scope, params::SigmoidRegression, x::Variable, p) = @with scope sigmoid(score(params, x, p))

cost(scope::Scope, params::SigmoidRegression, x::Variable, y) = @with scope Cost.bernoulli_cross_entropy_with_scores(score(params, x), y)
cost(scope::Scope, params::SigmoidRegression, x::Variable, y, p) = @with scope Cost.bernoulli_cross_entropy_with_scores(score(params, x, p), y)

weighted_cost(scope::Scope, params::SigmoidRegression, x::Variable, y, weight::Real) = @with scope Cost.bernoulli_cross_entropy_with_scores(score(params, x), y, weight)
weighted_cost(scope::Scope, params::SigmoidRegression, x::Variable, y, p, weight::Real) = @with scope Cost.bernoulli_cross_entropy_with_scores(score(params, x, p), y, weight)

