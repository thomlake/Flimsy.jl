
abstract Variable{T}

immutable GradVariable{T<:AbstractFloat} <: Variable{T}
    data::Matrix{T}
    grad::Matrix{T}
end

immutable DataVariable{T<:AbstractFloat} <: Variable{T}
    data::Matrix{T}
end

Input(x::Matrix) = DataVariable(x)

Input(x::Vector) = DataVariable(reshape(x, length(x), 1))

Base.size(x::Variable) = size(x.data)

Base.size(x::Variable, d::Integer) = size(x.data, d)

Base.eltype{T}(::Type{GradVariable{T}}) = T

Base.eltype{T}(::Type{DataVariable{T}}) = T

Base.eltype{V<:Variable}(x::V) = eltype(V)

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

function Base.show{V<:Variable}(io::IO, x::V)
    m, n = size(x)
    print(io, "$(m)x$(n) $V")
end

function Base.show{V<:Variable}(io::IO, xs::Vector{V})
    print(io, eltype(xs), "[", join(map(x-> "$(size(x,1))x$(size(x,2))", xs), ", "), "]")
end