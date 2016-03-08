"""
Wrapper for creating an Array of Flimsy variables. 
Uses information about the current scope to determine 
the correct eltype.
"""
function Sequence end

Sequence{R<:AbstractFloat}(scope::Scope, ::Type{R}, n::Int) = Array(DataVariable{R}, n)

Sequence{R<:AbstractFloat}(scope::GradScope, ::Type{R}, n::Int) = Array(GradVariable{R}, n)

Sequence(scope::Scope, n::Int) = Sequence(scope, Float64, n)
