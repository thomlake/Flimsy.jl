type NimMat{R<:FloatingPoint}
    x::Matrix{R}
    dx::Matrix{R}
end

Base.eltype{R}(::Type{NimMat{R}}) = R

Base.eltype{R}(mat::NimMat{R}) = R

Base.size(mat::NimMat) = size(mat.x)

Base.size(mat::NimMat, d::Integer) = size(mat.x, d)

NimMat(w::Matrix) = NimMat(w, zero(w))

NimMat(w::Vector) = NimMat(reshape(w, length(w), 1))

Base.zeros{R}(::Type{NimMat{R}}, d1::Integer, d2::Integer) = NimMat(zeros(R, d1, d2))

Base.zeros(::Type{NimMat}, d1::Integer, d2::Integer) = NimMat(zeros(d1, d2))

Base.zeros{R}(::Type{NimMat{R}}, d::Int) = NimMat(zeros(R, d))

Base.zeros(::Type{NimMat}, d::Int) = NimMat(zeros(d))

Base.zeros{R}(::Type{NimMat{R}}, t::Tuple) = NimMat(zeros(R, t))

Base.zeros(::Type{NimMat}, t::Tuple) = NimMat(zeros(t))

Base.zero{T}(mat::NimMat{T}) = zeros(NimMat{T}, size(mat))
