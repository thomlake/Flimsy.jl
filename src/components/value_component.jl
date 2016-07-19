immutable EmptyComponent <: Component end

immutable ValueComponent <: Component
    value::Variable
end

immutable VectorComponent <: Component
    values::Vector{Variable}
end
