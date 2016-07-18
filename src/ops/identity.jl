
Base.identity(scope::Scope, x::AbstractValue) = x

Base.identity(scope::RunScope, x::Variable) = Constant(x.data)
