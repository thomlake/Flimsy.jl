immutable LogisticRegression{T,M,N} <: Component
    w::Var{T,M,N}
    b::Var{T,M,1}
end

LogisticRegression(w::AbstractMatrix, b::AbstractVector) = LogisticRegression(Var(w), Var(b))

LogisticRegression(n_classes, n_features) = LogisticRegression(Gaussian(n_classes, n_features), Zeros(n_classes))

@Flimsy.component score(theta::LogisticRegression, x::Var) = affine(theta.w, x, theta.b)

@Flimsy.component predict(theta::LogisticRegression, x) = Flimsy.Extras.argmax(score(theta, x))

@Flimsy.component probs(theta::LogisticRegression, x) = softmax(score(theta, x))

@Flimsy.component function probs(theta::LogisticRegression, x, y)
    p = probs(theta, x)
    nll = Flimsy.Cost.cat(y, p)
    return nll, p
end
