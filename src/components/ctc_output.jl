
immutable CtcOutput{V<:Variable} <: LinearModel{V}
    w::V
    b::V
    blank::Int
    function CtcOutput(w::V, b::V, blank::Int)
        m, n = size(w)
        size(b) == (m, 1) || throw(DimensionMismatch("Bad size(b) == $(size(b)) != ($m, 1)"))
        return new(w, b, blank)
    end
end
CtcOutput{V<:Variable}(w::V, b::V, blank::Int) = CtcOutput{V}(w, b, blank)

function CtcOutput(n_output::Int, n_input::Int, blank::Int=1)
    return CtcOutput(w=rand(Normal(0, 0.01), n_output, n_input), b=zeros(n_output, 1), blank=blank)
end

@component score(params::CtcOutput, xs::Vector) = [score(params, x) for x in xs]

@component probs(params::CtcOutput, xs::Vector) = [softmax(y) for y in score(params, xs)]

@component predict(params::CtcOutput, xs::Vector) = [argmax(y) for y in score(params, xs)]

@component function cost(params::CtcOutput, xs::Vector, ys::Vector)
    return Cost.ctc_with_scores(score(params, xs), ys, params.blank)
end
