
function get_component_strings!(strings::Vector{AbstractString}, d::Dict, indent::Int=2)
    indent_string = repeat(" ", indent)
    for (k, v) in d
        if isa(v, Dict)
            push!(strings, string(indent_string, k, " => "))
            get_component_strings!(strings, v, indent + 2)
        elseif isa(v, Vector)
            push!(strings, string(indent_string, k, " => ", v))
        else
            m, n = size(v)
            push!(strings, string(indent_string, k, " => ", m, "x", n, " ", typeof(v)))
        end
    end
end

function Base.show{C<:Component}(io::IO, params::C, indent::Int=0)
    println(io, repeat(" ", indent), C.name, " =>")
    strings = AbstractString[]
    get_component_strings!(strings, convert(Dict, params), indent + 2)
    print(io, join(strings, "\n"))
end