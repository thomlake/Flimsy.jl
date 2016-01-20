
function compile(f::Function, params::Component, types::DataType...)
    gparams = GradComponent(params)
    gptr = pointer_from_objref(gparams)
    names = Symbol[]
    cargs = Any[]
    for typ in types
        name = gensym()
        push!(names, name)
        if typ <: Variable
            push!(cargs, :(GradIntput($name)))
        elseif typ <: AbstractVector && eltype(t) <: Variable
            push!(cargs, :([GradIntput(a) for a in $name]))
        else
            push!(cargs, name)
        end
    end
    println(cargs)
    sig = Expr(:tuple, names...)
    body = Expr(:call, symbol(f), :(unsafe_pointer_to_objref($gptr)), cargs...)
    anon = Expr(:(->), sig, body)
    expr = Expr(:macrocall, Symbol("@anon"), anon)
    return eval(macroexpand(expr))
end

# function compile(scope::DataScope, name::Symbol, args::OrderedDict{Symbol,DataType}; gradients::Bool=false)
#     if gradients
#         scope = GradScope(scope)
#     end
#     scope_ptr = pointer_from_objref(scope)
#     cargs = Any[]
#     for (arg, typ) in args
#         if typ <: Variable
#             push!(cargs, :(InputVariable($arg)))
#         elseif typ <: AbstractVector && eltype(t) <: Variable
#             push!(cargs, :( [InputVariable(unsafe_pointer_to_objref($scope_ptr), a) for a in $arg] ))
#         else
#             push!(cargs, arg)
#         end
#     end
#     sig = Expr(:tuple, keys(args)...)
#     body = Expr(:call, name, :(unsafe_pointer_to_objref($scope_ptr)), cargs...)
#     anon = Expr(:(->), sig, body)
#     expr = Expr(:macrocall, Symbol("@anon"), anon)
#     return eval(macroexpand(expr))
# end