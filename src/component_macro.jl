
type ParseError <: Exception
    msg::ASCIIString
end

Base.showerror(io::IO, e::ParseError) = print(io, "Flimsy.ParseError: ", e.msg)

"""Array of function names that Flimsy should not backpropage through"""
const BLACKLIST = [
    # :Variable,
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
    :isa,
    :push!,
    :append!,
    :rand,
    :if,
    :error,
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
    :Input,
    :bagofwords,
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
    :+=,
    :-=,
    :*=,
    :/=,
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
Recursively rewrite expr so all :call expression have scope as their first argument.

Example

    insert_scope(:scope, :(foo(a, b, c)))

Result

    foo(a, b, c) 
    == Expr(:call, :foo, :a, :b, :c)
    -> Expr(:call, :foo, scope, :a, :b, :c) 
    == foo(scope, a, b, c)

"""
function insert_scope(scope::Symbol, expr::Expr)
    head = expr.head
    args = expr.args
    newargs = Any[]

    # Special cases: line numbers and directives
    if head == :line
        return expr
    end

    if head == :call
        if !in(args[1], BLACKLIST)
            push!(newargs, shift!(args))
            push!(newargs, scope)
        end
    elseif !in(head, SUPPORTED_SYNTAX)
        throw(ParseError("Unsupported Expr: ($head, $args)"))
    end

    for arg in args
        if typeof(arg) <: Expr
            newarg = insert_scope(scope, arg)
            if newarg != nothing
                push!(newargs, newarg)
            end
        else
            push!(newargs, arg)
        end
    end
    return Expr(head, newargs...)
end

macro with(scope::Symbol, body::Expr)
    body = insert_scope(scope, body)
    return esc(body)
end
