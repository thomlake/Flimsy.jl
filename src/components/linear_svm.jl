
immutable LinearSVM{T,M,N} <: Component{T,M,N}
    w::Variable{T,M,N}
    b::Variable{T,M,1}
end

LinearSVM(w::AbstractMatrix, b::AbstractVector) = LinearSVM(Variable(w), Variable(b))

LinearSVM(n_classes, n_features) = LinearSVM(Gaussian(n_classes, n_features), Zeros(n_classes))

@flimsy score(theta::LinearSVM, x::Variable) = affine(theta.w, x, theta.b)

@flimsy predict(theta::LinearSVM, x) = Flimsy.Extras.argmax(score(theta, x))

@flimsy function predict(theta::LinearSVM, x, y, m::Real=1)
    p = score(theta, x)
    loss = Flimsy.Cost.margin(y, p, m)
    return loss, p
end
