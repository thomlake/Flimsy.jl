
type ComponentParseError <: Exception
    msg::ASCIIString
end

Base.showerror(io::IO, e::ComponentParseError) = print(io, "Flimsy.ComponentParseError: ", e.msg)

"""Array of function names that Flimsy should not backpropage through"""
const DEFAULT_BLACKLIST = [
    :Variable,
    :Array,
    :size,
    :length,
    :eachindex,
    :endof,
    :reverse,
    :enumerate,
    :zip,
    :collect,
    :print,
    :println,
    :eltype,
    :typeof,
    :push!,
    :append!,
    :rand,
    :if,
    :!,
    :+,
    :-,
    :*,
    :/,
    :^,
    :+=,
    :-=,
    :*=,
    :/=,
    :&&,
    :||,
    :argmax,
    :argmaxneq,   
]

"""Array of supported expression head elements"""
const SUPPORTED_SYNTAX = [
    :block,
    :tuple,
    :dict,
    :vect,
    :(=),
    :(:),
    :(=>),
    :(.),
    :quote,
    :ref,
    :comparison,
    :return,
    :for,
    :while,
    :comprehension,
    :typed_comprehension,
    :curly,
    :kw,
]

"""
Create a wrapper function from the signature of a component function
to handle running the model, computing gradients, managing scope, and running gc.

Input:

    f(c::Foo, x, y) 

Output:

    function f{C<:Foo}(__model__::Model{C}, x, y; grad::Bool=false, force_gc::Bool=false)
        reset!(__model__.scope)
        if force_gc || __model__.step == __model__.gc_step
            gc()
            __model__.step = 0
        else
            __model__.step += 1
        end
        result = f(grad ? __model__.gradscope : __model__.scope, __model__.component, x, y)
        grad && backprop!(__model__.gradscope)
        return result
    end
"""
function wrapped_component_function(signature::Expr)
    name = signature.args[1]
    arg1 = signature.args[2]
    arg1.head == :(::) || error("expected name::Type but got $arg1")
    component_type = arg1.args[2]
    # TODO: Figure out how to check if component_type is actuall a subtype of Component
    # eval(component_type) <: Component || error("first arg must be subtype of Component")
    model_signature = :($name{C<:$component_type}(__model__::Model{C}, $(signature.args[3:end]...); grad::Bool=false, force_gc::Bool=false))
    call_args = [isa(a, Symbol) ? a : a.args[1] for a in signature.args[3:end]]
    model_body = quote
        reset!(__model__.scope)
        if force_gc || __model__.step == __model__.gc_step
            gc()
            __model__.step = 0
        else
            __model__.step += 1
        end
        result = $name(grad ? __model__.gradscope : __model__.scope, __model__.component, $(call_args...))
        grad && backprop!(__model__.gradscope)
        return result
    end
    return Expr(:(=), model_signature, model_body)
end

"""
Add a Scope type as the first argument in a function signature.

    f(x, y) => f(__scope__::Scope, x, y)
"""
function signature_with_scope(signature::Expr)
    new_signature = deepcopy(signature)
    insert!(new_signature.args, 2, Expr(:(::), :__scope__, :Scope))
    return new_signature
end

"""
Recursively rewrite expr so all non-blacklisted 
:call expression have __scope___ as their first argument.

For example, the following statment is tranformed as follows
    foo(a, b, c) ==  Expr(:call, :foo, :a, :b, :c)
                 =>  Expr(:call, :foo, __scope__, :a, :b, :c)
                 ==  foo(__scope__, a, b, c)
"""
function insert_scope(expr::Expr, blacklist::Vector)
    head = expr.head
    args = expr.args
    newargs = Any[]

    # Special cases: line numbers and directives
    if head == :line
        return expr, blacklist
    elseif head == :macrocall && args[1] == symbol("@blacklist")
        if all(a -> isa(a, Symbol), args[2:end])
            return nothing, [blacklist..., args[2:end]...]
        else
            throw(ComponentParseError("Malformed @blacklist directive: $args"))
        end
    elseif head == :macrocall && args[1] == symbol("@similar_variable_type")
        if length(args) == 3 && isa(args[2], Symbol) && isa(args[3], Symbol)
            return :($(args[2]) = GradVariable{eltype($(args[3]))}), blacklist
        else
            throw(ComponentParseError("Malformed @vartype directive: $args"))
        end
    end

    if head == :call
        if !in(args[1], blacklist)
            push!(newargs, shift!(args))
            push!(newargs, :__scope__)
        end
    elseif in(head, blacklist)
        pass
    elseif !in(head, SUPPORTED_SYNTAX)
        throw(ComponentParseError("Unsupported Expr: ($head, $args)"))
    end
    
    inner_blacklist = deepcopy(blacklist)

    for arg in args
        if typeof(arg) <: Expr
            newarg, inner_blacklist = insert_scope(arg, inner_blacklist)
            if newarg != nothing
                push!(newargs, newarg)
            end
        else
            push!(newargs, arg)
        end
    end
    return Expr(head, newargs...), blacklist
end

function remove_directives(expr::Expr)
    head = expr.head
    args = expr.args
    newargs = Any[]

    if head == :macrocall && expr.args[1] == symbol("@blacklist")
        return nothing
    elseif head == :macrocall && args[1] == symbol("@similar_variable_type")
        if length(args) == 3 && isa(args[2], Symbol) && isa(args[3], Symbol)
            return :($(args[2]) = DataVariable{eltype($(args[3]))})
        else
            throw(ComponentParseError("Malformed @vartype directive: $args"))
        end
    end

    for arg in args
        if typeof(arg) <: Expr
            newarg = remove_directives(arg)
            if newarg != nothing
                push!(newargs, newarg)
            end
        else
            push!(newargs, arg)
        end
    end
    return Expr(head, newargs...)
end


function create_component_functions(f::Expr)
    f.head == :(=) || f.head == :function || throw(ComponentParseError("expected = or function but got $(f.head)"))
    ok = length(f.args) == 2 && f.args[1].head == :call && f.args[2].head == :block
    if !ok 
        throw(ComponentParseError("expected :call or :block but got $(map(x->x.head, f.args))"))
    end

    signature = f.args[1]
    body = f.args[2]

    new_signature = signature_with_scope(signature)
    new_body, _ = insert_scope(body, DEFAULT_BLACKLIST)
    f1 = Expr(f.head, new_signature, new_body)
    f2 = wrapped_component_function(signature)
    return Expr(:block, f1, f2)
end

macro component(x::Expr)
    y = create_component_functions(x)
    return esc(y)
end
