
Base.identity(scope::Scope, x::Variable) = x
Base.identity(scope::DataScope, x::GradVariable) = DataVariable(x.data)
