immutable LinearRegression{T,M,N} <: Component
    w::Var{T,M,N}
    b::Var{T,M,1}
end

LinearRegression(w::Matrix, b::Vector) = LinearRegression(Var(w), Var(b))

LinearRegression(n_classes, n_features) = LinearRegression(Gaussian(n_classes, n_features), Zeros(n_classes))

@Flimsy.component predict(theta::LinearRegression, x::Var) = affine(theta.w, x, theta.b)

@Flimsy.component function predict(theta::LinearRegression, x, y)
    p = predict(theta, x)
    nll = Flimsy.Cost.gauss(y, p)
    return nll, p
end
