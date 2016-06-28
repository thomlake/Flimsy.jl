immutable EmptyComponent <: Component end

immutable ValueComponent <: Component
    value::GradVariable
end

immutable VectorComponent <: Component
    values::Vector{GradVariable}
end
