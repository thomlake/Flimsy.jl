"""
Wrapper for creating an Array of Flimsy variables. 
Uses information about the current scope to determine 
the correct eltype.
"""
function Sequence end

Sequence(scope::Scope, n::Int) = Array(Constant, n)

Sequence(scope::GradScope, n::Int) = Array(Variable, n)

NTupleSequence(scope::Scope, m::Int, n::Int) = Array(NTuple{m,Constant}, n)

NTupleSequence(scope::GradScope, m::Int, n::Int) = Array(NTuple{m,Variable}, n)
