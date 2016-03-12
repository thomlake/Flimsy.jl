
"""
Convergence monitoring that triggers when 
`frustration` updates occur without imporvement.
"""
type Patience
    frustration::Int
    patience::Int
end

Patience(patience::Number=1) = Patience(0, patience)

function call(self::Patience, improved::Bool)
    if improved
        self.frustration = 0
    else
        self.frustration += 1
    end
    return self.frustration > self.patience
end
