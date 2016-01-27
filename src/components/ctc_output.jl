
immutable CTCOutput{V<:Variable} <: LinearModel{V}
    w::V
    b::V
    blank::Int
    function CTCOutput(w, b, blank)
        m, n = size(w)
        size(b) == (m, 1) || throw(DimensionMismatch("Bad size(b) == $(size(b)) != ($m, 1)"))
        return new(w, b, blank)
    end
end

function CTCOutput(n_output::Int, n_input::Int, blank::Int=1)
    return CTCOutput(w=rand(Normal(0, 0.01), n_output, n_input), b=zeros(n_output), blank=blank)
end

@component score(params::CTCOutput, xs::Vector) = [score(params, x) for x in xs]

@component probs(params::CTCOutput, xs::Vector) = [softmax(y) for y in score(params, xs)]

@component predict(params::CTCOutput, xs::Vector) = [argmax(y) for y in score(params, xs)]

@component function cost{I<:Integer}(params::CTCOutput, xs::Vector, ys::Vector{I})
    return Cost.ctc_with_scores(score(params, xs), ys, params.blank)
end
