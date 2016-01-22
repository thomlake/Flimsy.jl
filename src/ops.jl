#
# Operator Implementations
# ------------------------
# 
# At least two versions of an operator f must be implemented. The **Base Operator**
# always return Variables with type T<:DataVariable (typically used at test time).
# The **Grad Operator** promotes output Variables to type T<:GradVariable when 
# necessary (typically used at train time). For the later, staged (@generated) functions 
# are useful for generating efficient code which is type stable and does not incur 
# repeated runtime type checking costs.
#
# Additionally a closure type (a subtype of ReverseOperation) must be implemented for 
# each operation to be backpropagated through. This type should be instantiated in the 
# Grad Operator's body and pushed onto the CallbackStack. A ReverseOperation stores references 
# to any variables necessary for computing gradients and is responsible for setting each variable's 
# grad field to the gradient of that value in its call function.
#
# The implementations are as as follows:
#
# The Base Operator
# f: X -> Y
# - If f returns an element y s.t. typeof(y)<:Variable then typeof(y)<:DataVariable.
# - This requirement eliminates cascading promotion of outputs to GradVariable 
#   (regardless of the types of inputs) when gradients are not needed, i.e., at test time.  
#   Not enforcing this would either incur increased memory requirements from allocating storage
#   for gradients that would never be used, or require storing 2 sets of parameters, one with and one 
#   without storage for gradients.
#
# The Grad Operator
# f: CallbackStack * X -> Y
# - The CallbackStack simulatneously indicates that we want gradients and stores ReverseOperations
#   for performing backpropagation.
# - In general if any Variable x in X is a subtype of GradVariable, then all Variables in Y should
#   also have type GradVariable. This is because their gradient will be required for computing
#   gradients with respect to input variables X.
#
export anygrads

export sigmoid,
       relu,
       wta,
       softmax,
       plus,
       minus,
       mult,
       linear,
       affine,
       decat,
       concat

export OperationError

type OperationError <: Exception
    msg::ASCIIString
end

Base.showerror(io::IO, e::OperationError) = print(io, "Flimsy.OperationError: ", e.msg)

function anygrads(ts::DataType...)
    for t in ts
        if t <: GradVariable
            return true
        end
    end
    return false
end

anygrads(ts::Vector) = anygrads(ts...)

include("ops/identity.jl")
include("ops/tanh.jl")
include("ops/sigmoid.jl")
include("ops/relu.jl")
include("ops/wta.jl")
include("ops/softmax.jl")
include("ops/softmax_vector.jl")
include("ops/plus.jl")
include("ops/minus_scalar.jl")
include("ops/minus.jl")
include("ops/mult_scalar.jl")
include("ops/mult.jl")
include("ops/linear.jl")
include("ops/affine.jl")
include("ops/decat.jl")
include("ops/concat.jl")
