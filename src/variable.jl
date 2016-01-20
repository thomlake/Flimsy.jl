
abstract Variable{T}

type GradVariable{T<:AbstractMatrix} <: Variable{T}
    data::T
    grad::T
end

typealias GradInput GradVariable

call{V<:GradVariable}(::Type{V}, x::AbstractMatrix) = GradVariable(x, zero(x))

call{V<:GradVariable}(::Type{V}, x::AbstractVector) = GradVariable(reshape(x, length(x), 1))

call{V<:GradVariable}(::Type{V}, x::Real) = GradVariable([x])

type DataVariable{T<:AbstractMatrix} <: Variable{T}
    data::T
end

# GradVariable{V<:DataVariable}(x::V) = GradVariable(x.data, zero(x.data))
# call{G<:GradVariable,D<:DataVariable}(::Type{G}, x::D) = GradVariable(x.data, zero(x.data))

typealias Input{T} DataVariable{T}

call{V<:DataVariable}(::Type{V}, x::AbstractVector) = DataVariable(reshape(x, length(x), 1))

call{V<:DataVariable}(::Type{V}, x::Real) = DataVariable([x])

Base.size(x::Variable) = size(x.data)

Base.size(x::Variable, d::Integer) = size(x.data, d)

Base.eltype{V<:Variable}(::Type{V}) = eltype(V)

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

