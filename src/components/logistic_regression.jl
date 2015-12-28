
immutable LogisticRegression{T,M,N} <: Component{T,M,N}
    w::Variable{T,M,N}
    b::Variable{T,M,1}
end

LogisticRegression(n_out::Int, n_in::Int) = LogisticRegression(
    w=glorot(n_out, n_in), 
    b=zeros(n_out),
)

@flimsy score(theta::LogisticRegression, x::Variable) = affine(theta.w, x, theta.b)

@flimsy predict(theta::LogisticRegression, x) = Flimsy.Extras.argmax(score(theta, x))

@flimsy probs(theta::LogisticRegression, x) = softmax(score(theta, x))

@flimsy cost(theta::LogisticRegression, x, y) = Flimsy.Cost.cat(y, probs(theta, x))
