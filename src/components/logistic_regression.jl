immutable LogisticRegression{T,M,N} <: Component
    w::Variable{T,M,N}
    b::Variable{T,M,1}
end

LogisticRegression(w::AbstractMatrix, b::AbstractVector) = LogisticRegression(Variable(w), Variable(b))

LogisticRegression(n_classes, n_features) = LogisticRegression(Gaussian(n_classes, n_features), Zeros(n_classes))

@flimsy score(theta::LogisticRegression, x::Variable) = affine(theta.w, x, theta.b)

@flimsy predict(theta::LogisticRegression, x) = Flimsy.Extras.argmax(score(theta, x))

@flimsy probs(theta::LogisticRegression, x) = softmax(score(theta, x))

@flimsy function probs(theta::LogisticRegression, x, y)
    p = probs(theta, x)
    nll = Flimsy.Cost.cat(y, p)
    return nll, p
end
