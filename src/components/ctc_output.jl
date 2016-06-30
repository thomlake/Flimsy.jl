
immutable CtcOutput <: LinearModel
    w::GradVariable
    b::GradVariable
    blank::Int
    function CtcOutput(w::GradVariable, b::GradVariable, blank::Int)
        m, n = size(w)
        size(b) == (m, 1) || throw(DimensionMismatch("Bad size(b) == $(size(b)) != ($m, 1)"))
        return new(w, b, blank)
    end
end

function CtcOutput(n_output::Int, n_input::Int, blank::Int=1)
    return CtcOutput(w=rand(Normal(0, 0.01), n_output, n_input), b=zeros(n_output, 1), blank=blank)
end

score(scope::Scope, params::CtcOutput, xs::Vector) = @with scope [score(params, x) for x in xs]

probs(scope::Scope, params::CtcOutput, xs::Vector) = @with scope [softmax(y) for y in score(params, xs)]

predict(scope::Scope, params::CtcOutput, xs::Vector) = @with scope [argmax(y) for y in score(params, xs)]

cost(scope::Scope, params::CtcOutput, xs::Vector, ys::Vector) = @with scope Cost.ctc_with_scores(score(params, xs), ys, params.blank)
