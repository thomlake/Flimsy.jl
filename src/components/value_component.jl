
immutable ValueComponent{V<:Variable} <: Component{V}
    value::V
end

immutable EmptyComponent <: Component end
