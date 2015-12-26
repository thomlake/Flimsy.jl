using Flimsy
using Base.Test

for (M, N) in [(1, 2), (2, 3), (3,1)]
    for T in [Float64, Float32, Float16]
        x = zeros(Variable, T, M, N)
        @test typeof(x) <: Variable{Array{T, 2},M,N}
        @test typeof(x) == Variable{Array{T, 2},M,N}
        @test eltype(x) == T
        @test all(x.data .== 0)
        @test all(x.grad .== 0)
        @test size(x) == (M, N)
        @test size(x, 1) == M
        @test size(x, 2) == N
    end
end

# forbidden sizes
for sizeargs in [(1, 2, 3), (5, 4, 3, 2)]
    @test_throws MethodError zeros(Variable, sizeargs...)
end
