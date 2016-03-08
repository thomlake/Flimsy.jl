
immutable ValueComponent{V<:Variable} <: Component{V}
    value::V
end

immutable VectorComponent{V<:Variable} <: Component{V}
    values::Vector{V}
end

immutable EmptyComponent <: Component end
