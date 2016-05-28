
abstract Variable

immutable GradVariable <: Variable
    data::Matrix{FloatX}
    grad::Matrix{FloatX}
end

immutable DataVariable <: Variable
    data::Matrix{FloatX}
end

Input(x::Matrix) = DataVariable(x)

Input(x::Vector) = DataVariable(reshape(x, length(x), 1))

Variable(scope::Scope, x::Variable) = DataVariable(x.data)

Variable(scope::GradScope, x::DataVariable) = GradVariable(x.data, zero(x.data))

Variable(scope::GradScope, x::GradVariable) = x

Base.size(x::Variable) = size(x.data)

Base.size(x::Variable, d::Integer) = size(x.data, d)

Base.eachindex(x::Variable) = eachindex(x.data)

Base.zero(scope::Scope, x::Variable) = DataVariable(zero(x.data))

Base.zero(scope::GradScope, x::Variable) = GradVariable(zero(x.data), zero(x.data))

Base.zeros(scope::Scope, m::Int, n::Int) = DataVariable(zeros(m, n))

Base.zeros(scope::GradScope, m::Int, n::Int) = GradVariable(zeros(m, n), zeros(m, n))

Base.ones(scope::Scope, m::Int, n::Int) = DataVariable(ones(m, n))

Base.ones(scope::GradScope, m::Int, n::Int) = GradVariable(ones(m, n), ones(m, n))

function is_matrix(x::Variable)
    m, n = size(x)
    return m > 1 && n > 1
end

function is_column_vector(x::Variable)
    m, n = size(x)
    return m > 1 && n == 1
end

function is_row_vector(x::Variable)
    m, n = size(x)
    return m == 1 && n > 1
end

function is_scalar(x::Variable)
    m, n = size(x)
    return m == 1 && n == 1
end

function Base.show{V<:Variable}(io::IO, x::V)
    m, n = size(x)
    print(io, "$(m)x$(n) $V")
end

function Base.show{V<:Variable}(io::IO, xs::Vector{V})
    print(io, eltype(xs), "[", join(map(x-> "$(size(x,1))x$(size(x,2))", xs), ", "), "]")
end