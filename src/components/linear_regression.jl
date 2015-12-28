immutable LinearRegression{T,M,N} <: Component{T,M,N}
    w::Variable{T,M,N}
    b::Variable{T,M,1}
end

LinearRegression(n_classes, n_features) = LinearRegression(
    w=glorot(n_classes, n_features), 
    b=zeros(n_classes),
)

@flimsy predict(theta::LinearRegression, x::Variable) = affine(theta.w, x, theta.b)

@flimsy cost(theta::LinearRegression, x, y) = Flimsy.Cost.gauss(y, predict(theta, x))
