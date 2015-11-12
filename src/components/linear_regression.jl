immutable LinearRegression{T,M,N} <: Component
    w::Variable{T,M,N}
    b::Variable{T,M,1}
end

LinearRegression(w::Matrix, b::Vector) = LinearRegression(Variable(w), Variable(b))

LinearRegression(n_classes, n_features) = LinearRegression(Gaussian(n_classes, n_features), Zeros(n_classes))

@flimsy predict(theta::LinearRegression, x::Variable) = affine(theta.w, x, theta.b)

@flimsy function predict(theta::LinearRegression, x, y)
    p = predict(theta, x)
    nll = Flimsy.Cost.gauss(y, p)
    return nll, p
end
