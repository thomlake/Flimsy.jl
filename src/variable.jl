
abstract AbstractValue

type Variable <: AbstractValue
    data::Matrix{FloatX}
    grad::Matrix{FloatX}
end

Variable(data::Matrix) = Variable(data, zero(data))

type Constant <: AbstractValue
    data::Matrix{FloatX}
end

Input(x::Matrix) = Constant(x)

Input(x::Vector) = Constant(reshape(x, length(x), 1))

Value(scope::Scope, x::AbstractValue) = Constant(x.data)

Value(scope::GradScope, x::AbstractValue) = x

Base.size(x::AbstractValue) = size(x.data)

Base.size(x::AbstractValue, d::Integer) = size(x.data, d)

Base.eachindex(x::AbstractValue) = eachindex(x.data)

Base.zero(scope::Scope, x::AbstractValue) = Constant(zero(x.data))

Base.zero(scope::GradScope, x::AbstractValue) = Variable(zero(x.data), zero(x.data))

Base.zeros(scope::Scope, m::Int, n::Int) = Constant(zeros(m, n))

Base.zeros(scope::GradScope, m::Int, n::Int) = Variable(zeros(m, n), zeros(m, n))

Base.ones(scope::Scope, m::Int, n::Int) = Constant(ones(m, n))

Base.ones(scope::GradScope, m::Int, n::Int) = Variable(ones(m, n), ones(m, n))

function is_matrix(x::AbstractValue)
    m, n = size(x)
    return m > 1 && n > 1
end

function is_column_vector(x::AbstractValue)
    m, n = size(x)
    return m > 1 && n == 1
end

function is_row_vector(x::AbstractValue)
    m, n = size(x)
    return m == 1 && n > 1
end

function is_scalar(x::AbstractValue)
    m, n = size(x)
    return m == 1 && n == 1
end

function Base.show{V<:AbstractValue}(io::IO, x::V)
    m, n = size(x)
    print(io, "$(m)x$(n) $V")
end

function Base.show{V<:AbstractValue}(io::IO, xs::Vector{V})
    print(io, eltype(xs), "[", join(map(x-> "$(size(x,1))x$(size(x,2))", xs), ", "), "]")
end