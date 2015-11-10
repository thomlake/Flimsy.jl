immutable LogisticRegression{T,M,N} <: Component
    w::Var{T,M,N}
    b::Var{T,M,1}
end

LogisticRegression(w::Matrix, b::Vector) = LogisticRegression(Var(w), Var(b))

LogisticRegression(n_classes, n_features) = LogisticRegression(Gaussian(n_classes, n_features), Zeros(n_classes))

@Nimble.component score(theta::LogisticRegression, x::Var) = affine(theta.w, x, theta.b)

@Nimble.component predict(theta::LogisticRegression, x) = Nimble.Extras.argmax(score(theta, x))

@Nimble.component probs(theta::LogisticRegression, x) = softmax(score(theta, x))

@Nimble.component function probs(theta::LogisticRegression, x, y)
    p = probs(theta, x)
    nll = Nimble.Cost.cat(y, p)
    return nll, p
end
