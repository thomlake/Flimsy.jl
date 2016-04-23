
type ShuffledIter{T}
    data::Vector{T}
    indices::Vector{Int}
end
ShuffledIter(data::Vector) = ShuffledIter(data, collect(1:length(data)))

Base.start(iter::ShuffledIter) = (shuffle!(iter.indices); 1)
Base.next{T}(iter::ShuffledIter{T}, i) = (iter.data[iter.indices[i]], i + 1)
Base.done(iter::ShuffledIter, i) = i > length(iter.indices)