
immutable ValueComponent{V<:Variable} <: Component
    value::V
end

immutable EmptyComponent <: Component end
