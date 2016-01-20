
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
Insert an AbstractScope as the first argument in a function signature.

    f(x, y) => f(__callback_stack__::CallbackStack, x, y)
"""
function signature_with_stack(signature::Expr)
    new_signature = deepcopy(signature)
    insert!(new_signature.args, 2, Expr(:(::), :__callback_stack__, :CallbackStack))
    return new_signature
end

"""
Recursively rewrite expr so all non-blacklisted 
:call expression have __callback_stack___ as their first argument.

For example, the following statment is tranformed as follows
    foo(a, b, c) ==  Expr(:call, :foo, :a, :b, :c)
                 =>  Expr(:call, :foo, __callback_stack__, :a, :b, :c)
                 ==  foo(__callback_stack__, a, b, c)
"""
function insert_stack(expr::Expr, blacklist::Vector)
    head = expr.head
    args = expr.args
    newargs = Any[]

    # Special cases: line numbers and directives
    if head == :line
        return expr, blacklist
    elseif head == :macrocall && expr.args[1] == symbol("@blacklist")
        return nothing, [blacklist..., expr.args[2:end]...]
    end

    if head == :call
        if !in(args[1], blacklist)
            push!(newargs, shift!(args))
            push!(newargs, :__callback_stack__)
        end
    elseif in(head, blacklist)
        pass
    elseif !in(head, SUPPORTED_SYNTAX)
        throw(ComponentParseError("Unsupported Expr: ($head, $args)"))
    end
    
    inner_blacklist = deepcopy(blacklist)

    for arg in args
        if typeof(arg) <: Expr
            newarg, inner_blacklist = insert_stack(arg, inner_blacklist)
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

    stack_signature = signature_with_stack(signature)
    stack_body, _ = insert_stack(deepcopy(body), DEFAULT_BLACKLIST)
    g1 = Expr(f.head, signature, remove_directives(body))
    g2 = Expr(f.head, stack_signature, stack_body)
    return Expr(:block, g1, g2)
end

macro component(x::Expr)
    y = create_component_functions(x)
    return esc(y)
end
