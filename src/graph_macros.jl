
macro backprop(expr::Expr)
    if expr.head != :call
        throw(ArgumentError("expected: call, got: ", expr.head))
    end
    insert!(expr.args, 2, :gradscope)
    new_expr = quote
        gradscope = GradScope()
        output = $expr
        backprop!(gradscope)
        output
    end
    return esc(new_expr)
end

macro run(expr::Expr)
    if expr.head != :call
        throw(ArgumentError("expected :call, got ", expr.head))
    end
    insert!(expr.args, 2, :(RunScope()))
    return esc(expr)
end