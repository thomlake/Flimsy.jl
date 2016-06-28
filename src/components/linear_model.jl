
"""Abstract Linear Model."""
abstract LinearModel <: Component

function call{C<:LinearModel}(::Type{C}, n_output::Int, n_input::Int)
    return C(w=rand(Normal(0, 0.01), n_output, n_input), b=zeros(n_output, 1))
end

score(scope::Scope, params::LinearModel, x::Variable) = @with scope affine(params.w, x, params.b)
score(scope::Scope, params::LinearModel, x::Variable, p::AbstractFloat) = @with scope affine(params.w, dropout(x, p), params.b)

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
predict(scope::Scope, params::LinearRegression, x::Variable, p::AbstractFloat) = @with scope score(params, dropout!(x, p))

cost(scope::Scope, params::LinearRegression, x::Variable, y) = @with scope Cost.mse(score(params, x), y)
cost(scope::Scope, params::LinearRegression, x::Variable, y, p::AbstractFloat) = @with scope Cost.mse(score(params, dropout!(x, p)), y)

weighted_cost(scope::Scope, params::LinearRegression, x::Variable, y, weight::AbstractFloat) = @with scope Cost.mse(score(params, x), y, weight)
weighted_cost(scope::Scope, params::LinearRegression, x::Variable, y, weight::AbstractFloat, p::AbstractFloat) = @with scope Cost.mse(score(params, dropout!(x, p)), y, weight)

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
probs(scope::Scope, params::SoftmaxRegression, x::Variable) = @with scope softmax(score(params, x))
cost(scope::Scope, params::SoftmaxRegression, x::Variable, y) = @with scope Cost.categorical_cross_entropy_with_scores(score(params, x), y)
cost(scope::Scope, params::SoftmaxRegression, x::Variable, y, p::AbstractFloat) = @with scope Cost.categorical_cross_entropy_with_scores(score(params, dropout!(x, p)), y)

weighted_cost(scope::Scope, params::SoftmaxRegression, x::Variable, y, weight::Real) = @with Cost.categorical_cross_entropy_with_scores(score(params, x), y, weight)
weighted_cost(scope::Scope, params::SoftmaxRegression, x::Variable, y, weight::AbstractFloat, p::AbstractFloat) = @with scope Cost.categorical_cross_entropy_with_scores(score(params, dropout!(x, p)), y, weight)

"""
Sigmoid Regression Component.
    
    y[i]|x ~ Bernoulli(f(x)[i])
    f(x) = sigmoid(w * x + b)
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

@comp function predict(params::SigmoidRegression, x::Variable, threshold::Real=0.5)
    @blacklist log
    return score(params, x).data .> log(threshold / (1 - threshold))
end
@comp probs(params::SigmoidRegression, x::Variable) = sigmoid(score(params, x))
@comp cost(params::SigmoidRegression, x::Variable, y) = Cost.bernoulli_cross_entropy_with_scores(score(params, x), y)
@comp cost(params::SigmoidRegression, x::Variable, y, weight::Real) = Cost.bernoulli_cross_entropy_with_scores(score(params, x), y, weight)

