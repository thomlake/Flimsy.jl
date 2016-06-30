
type ShuffledIter{T}
    items::Vector{T}
    indices::Vector{Int}
end

shuffled(items::Vector) = ShuffledIter(items, collect(1:length(items)))
Base.start(iter::ShuffledIter) = (shuffle!(iter.indices); 1)
Base.next{T}(iter::ShuffledIter{T}, i::Int) = (convert(T, iter.items[iter.indices[i]]), i + 1)
Base.done(iter::ShuffledIter, i::Int) = i > length(iter.items)
Base.length(iter::ShuffledIter) = length(iter.items)
Base.eltype{T}(::Type{ShuffledIter{T}}) = T
