immutable CTCOutput{T,M,N} <: Component{T,M,N}
    w::Variable{T,M,N}
    b::Variable{T,M,1}
    blank::Int
end

CTCOutput(w::Matrix, b::Vector, blank::Int=1) = CTCOutput(Variable(w), Variable(b), blank)

CTCOutput(n_classes, n_features, blank::Int=1) = CTCOutput(Orthonormal(n_classes, n_features), Zeros(n_classes), blank)

@flimsy score(theta::CTCOutput, xs::Vector) = [affine(theta.w, xs[t], theta.b) for t = 1:length(xs)]

@flimsy probs(theta::CTCOutput, xs::Vector) = [softmax(y) for y in score(theta, xs)]

@flimsy predict(theta::CTCOutput, xs::Vector) = [Flimsy.Extras.argmax(y) for y in score(theta, xs)]

@flimsy cost(theta::CTCOutput, xs::Vector, ys::Vector{Int}) = Flimsy.Cost.ctc(ys, score(theta, xs), theta.blank)
