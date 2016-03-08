
Base.eltype{V<:Variable}(c::Component{V}) = eltype(V)

Base.eltype{V<:Variable}(::Type{Component{V}}) = eltype(V)

vartype{V<:Variable}(scope::Scope, c::Component{V}) = DataVariable{eltype(V)}

vartype{V<:Variable}(scope::Scope, ::Type{Component{V}}) = DataVariable{eltype(V)}

vartype{V<:Variable}(scope::GradScope, c::Component{V}) = GradVariable{eltype(V)}

vartype{V<:Variable}(scope::GradScope, ::Type{Component{V}}) = GradVariable{eltype(V)}

function show_component(io::IO, d::Dict, indent::Int=2)
    indent_string = repeat(" ", indent)
    for (k, v) in d
        if isa(v, Dict)
            println(io, indent_string, k, " => ")
            show_component(io, v, indent + 2)
        elseif isa(v, Vector)
            println(io, indent_string, k, " => ", v)
        else
            m, n = size(v)
            println(io, indent_string, k, " => ", m, "x", n, " ", typeof(v))
        end
    end
end

function Base.show{C<:Component}(io::IO, params::C, indent::Int=0)
    println(io, repeat(" ", indent), C.name)
    show_component(io, convert(Dict, params), indent + 2)
end