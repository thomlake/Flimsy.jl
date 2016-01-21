
export sigmoid,
       relu,
       wta,
       softmax,
       minus,
       linear,
       affine,
       decat,
       concat

export OperationError

type OperationError <: Exception
    msg::ASCIIString
end

Base.showerror(io::IO, e::OperationError) = print(io, "Flimsy.OperationError: ", e.msg)

include("ops/identity.jl")
include("ops/tanh.jl")
include("ops/sigmoid.jl")
include("ops/relu.jl")
include("ops/wta.jl")
include("ops/softmax.jl")
include("ops/softmax_vector.jl")
include("ops/sum.jl")
include("ops/scalar_minus.jl")
include("ops/minus.jl")
include("ops/scalar_prod.jl")
include("ops/prod.jl")
include("ops/linear.jl")
include("ops/affine.jl")
include("ops/decat.jl")
include("ops/concat.jl")