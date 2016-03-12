
macro flimsy_inbounds(expr::Expr)
    if FLIMSY_CONFIG["always_check_bounds"]
        return expr
    else
        return esc(:(@inbounds $expr))
    end
end
