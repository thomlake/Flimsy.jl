immutable LinearRegression{T,M,N} <: Component{T,M,N}
    w::Variable{T,M,N}
    b::Variable{T,M,1}
end

LinearRegression(w::AbstractArray, b::AbstractArray) = LinearRegression(Variable(w), Variable(b))

LinearRegression(n_classes, n_features) = LinearRegression(Gaussian(n_classes, n_features), Zeros(n_classes))

@flimsy predict(theta::LinearRegression, x::Variable) = affine(theta.w, x, theta.b)

@flimsy cost(theta::LinearRegression, x, y) = Flimsy.Cost.gauss(y, predict(theta, x))
