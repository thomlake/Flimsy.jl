immutable CTCOutput{T,M,N} <: Component
    w::Var{T,M,N}
    b::Var{T,M,1}
    blank::Int
end

CTCOutput(w::Matrix, b::Vector, blank::Int=1) = CTCOutput(Var(w), Var(b), blank)

CTCOutput(n_classes, n_features, blank::Int=1) = CTCOutput(Orthonormal(n_classes, n_features), Zeros(n_classes), blank)

@Flimsy.component score(theta::CTCOutput, xs::Vector) = [affine(theta.w, xs[t], theta.b) for t = 1:length(xs)]

@Flimsy.component probs(theta::CTCOutput, xs::Vector) = [softmax(y) for y in score(theta, xs)]

@Flimsy.component predict(theta::CTCOutput, xs::Vector) = [Flimsy.Extras.argmax(y) for y in score(theta, xs)]

@Flimsy.component function probs(theta::CTCOutput, xs::Vector, ys::Vector{Int})
    os = score(theta, xs)
    nll = Flimsy.Cost.ctc(ys, os, theta.blank)
    return nll
end
