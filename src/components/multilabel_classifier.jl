
immutable MultilabelClassifier{T,M,N} <: Component{T,M,N}
    w::Variable{T,M,N}
    b::Variable{T,M,1}
end

MultilabelClassifier(w::AbstractArray, b::AbstractArray) = MultilabelClassifier(Variable(w), Variable(b))

MultilabelClassifier(n_classes, n_features) = MultilabelClassifier(Gaussian(n_classes, n_features), Zeros(n_classes))

@flimsy score(theta::MultilabelClassifier, x::Variable) = affine(theta.w, x, theta.b)

@flimsy probs(theta::MultilabelClassifier, x) = sigmoid(score(theta, x))

@flimsy predict(theta::MultilabelClassifier, x, t::AbstractFloat=0.5) = map(Bool, threshold(probs(theta, x), t).data)

@flimsy cost(theta::MultilabelClassifier, x, y) = Flimsy.Cost.bern(y, probs(theta, x))
