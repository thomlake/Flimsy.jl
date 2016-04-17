"""
Wrapper for creating an Array of Flimsy variables. 
Uses information about the current scope to determine 
the correct eltype.
"""
function Sequence end

Sequence(scope::Scope, n::Int) = Array(DataVariable, n)

Sequence(scope::GradScope, n::Int) = Array(GradVariable, n)
