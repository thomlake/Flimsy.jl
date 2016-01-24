
abstract Variable{T}

type GradVariable{T<:AbstractMatrix} <: Variable{T}
    data::T
    grad::T
end

GradVariable{T}(x::Matrix{T}) = GradVariable(x, zero(x))

GradVariable{T}(x::Vector{T}) = GradVariable(reshape(x, length(x), 1))

GradVariable(x::Real) = GradVariable([x])


type DataVariable{T<:AbstractMatrix} <: Variable{T}
    data::T
end

DataVariable{T}(x::Vector{T}) = DataVariable(reshape(x, length(x), 1))

DataVariable(x::Real) = DataVariable([x])

Input(x) = DataVariable(x)

Base.size(x::Variable) = size(x.data)

Base.size(x::Variable, d::Integer) = size(x.data, d)

Base.eltype{T}(::Type{GradVariable{T}}) = T

Base.eltype{T}(::Type{DataVariable{T}}) = T

Base.eltype{V<:Variable}(x::V) = eltype(V)

Base.zero{V<:Variable}(x::V) = V(zero(x.data))

Base.zeros{V<:Variable,R}(::Type{V}, ::Type{R}, dims) = V(zeros(R, dims))

Base.zeros{V<:Variable}(::Type{V}, dims) = V(zeros(dims))

Base.eachindex(x::Variable) = eachindex(x.data)

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

