using Flimsy
import Flimsy.Components: Component, score, cost
using AliasTables
using IndexedArrays
using ProgressMeter
using Distances

const TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
const TEXT8_ZIP = "./text8.zip"
const TEXT8 = "./text8"
const PARAM_FILE = "./params"
const VOCAB_FILE = "./vocab"
const MIN_WORD_FREQ = 20
const PROB_POWER = 0.75
const CONTEXT_SIZE = 5
const EMBED_DIM = 50
const BATCH_SIZE = 50
const OPT_ALG = GradientDescent
const LEARNING_RATE = 0.01
const DECAY = 0.9
const CLIP = 40.0
const CLIPPING_TYPE = "none"

if !isfile(TEXT8)
    println("Downloading $TEXT8_URL")
    download(TEXT8_URL, TEXT8_ZIP)
    run(`unzip $TEXT8_ZIP`)
end

immutable Word2Vec <: Component
    W::Variable
    U::Variable
end

Word2Vec(embed_dim::Int, vocab_size::Int) = Word2Vec(
    W=rand(Normal(0, 0.01), embed_dim, vocab_size),
    U=rand(Normal(0, 0.01), embed_dim, vocab_size),
)

score(scope::Scope, θ::Word2Vec, x, y) = @with scope dot(linear(θ.W, x), linear(θ.U, y))

cost(scope::Scope, θ::Word2Vec, x, y, label) = @with scope Cost.bernoulli_cross_entropy_with_scores(score(θ, x, y), label)

function prepare_data()
    text = open(readall, PATH_TO_TEXT8)
    words = split(text, " ")
    println("Number of words => ", length(words))

    word_counts = Dict{ASCIIString,Int}()
    @showprogress 1 "Computing word frequencies..." for word in words
        word_counts[word] = get(word_counts, word, 0) + 1
    end

    words_keep = ASCIIString[]
    @showprogress 1 "Computing vocab..." for (word, count) in word_counts
        count >= MIN_WORD_FREQ && push!(words_keep, word)
    end
    println("Keeping ", length(words_keep), " of ", length(word_counts), " unique words")
    vocab = IndexedArray(words_keep)

    output = Int[]
    unigram_counts = zeros(Int, length(vocab))
    unigram_total = 0
    @showprogress 1 "Filtering input data..." for word in words
        i = get(vocab.lookup, word, 0)
        if i > 0 
            push!(output, i)
            unigram_counts[i] += 1
            unigram_total += 1
        end
    end
    println("Number of output words => ", length(output))
    
    probs = unigram_counts / unigram_total
    probs_pow = probs.^PROB_POWER
    table = AliasTable(probs_pow / sum(probs_pow))
    return output, vocab, table
end

function rand_index_in_context(i, max_len)
    r = max(1, i - CONTEXT_SIZE): min(max_len, i + CONTEXT_SIZE)
    for i = 1:100
        j = rand(r)
        i != j && return j
    end
    error("Max sampling iterations exceeded")
end

function train()
    data, vocab, table = prepare_data()
    indices = collect(1:length(data))
    params = Word2Vec(EMBED_DIM, length(vocab))
    opt = optimizer(OPT_ALG, params, learning_rate=LEARNING_RATE, decay=DECAY, clip=CLIP, clipping_type=CLIPPING_TYPE)
    # x_batch = [[0]  for i = 1:BATCH_SIZE]::Vector{Vector{Int}}
    # y_batch = [[0]  for i = 1:BATCH_SIZE]::Vector{Vector{Int}}
    # pos_lbl = fill(true, 1, BATCH_SIZE)
    # neg_lbl = fill(false, 1, BATCH_SIZE)
    # for epoch = 1:5
    #     nll = 0.0
    #     shuffle!(indices)
    #     @showprogress 60 "Epoch $epoch" for i = 1:BATCH_SIZE:length(data)
    #         # Positive
    #         for j = 1:BATCH_SIZE
    #             k = indices[i + j]
    #             x_batch[j][1] = data[k]
    #             y_batch[j][1] = data[rand_index_in_context(k, length(data))]
    #         end
    #         nll += @backprop cost(params, x_batch, y_batch, pos_lbl)
            
    #         # Negative
    #         for j = 1:BATCH_SIZE
    #             x_batch[j][1] = data[indices[i + j]]
    #             y_batch[j][1] = rand(table)
    #         end
    #         nll += @backprop cost(params, x_batch, y_batch, neg_lbl)

    #         # Update Parameters
    #         Flimsy.update!(opt)
    #     end
    #     println("avg nll = $(nll / length(data))")
    #     gc()
    for epoch = 1:5
        nll = 0.0
        shuffle!(indices)
        @showprogress 60 "Epoch $epoch..." for i = 1:length(data)
            k = indices[i]
            x = data[k]
            
            # Positive
            y = data[rand_index_in_context(k, length(data))]
            nll += @backprop cost(params, x, y, true)
            
            # Negative
            y = rand(table)
            nll += @backprop cost(params, x, y, false)

            # Update Parameters
            i % BATCH_SIZE == 0 && Flimsy.update!(opt)
        end
        println("avg nll = $(nll / length(data))")
        gc()
    end
    Flimsy.save(PARAM_FILE, params)
    fp = open(VOCAB_FILE, "w")
    for i = 1:length(vocab)
        println(fp, i, ",", vocab[i])
    end
    close(fp)
end

function interact()
    word2idx = open(VOCAB_FILE) do fp
        word2idx = Dict{ASCIIString,Int}()
        for line in readlines(fp)
            i, w = split(strip(line), ",")
            word2idx[w] = parse(Int, i)
        end
        word2idx
    end
    idx2word = Array(ASCIIString, length(word2idx))
    for (w, i) in word2idx
        idx2word[i] = w
    end

    params = Flimsy.restore(PARAM_FILE)
    embeddings = params.W.data
    while true
        print("word => ")
        w_i = lowercase(strip(readline(STDIN)))
        i = get(word2idx, w_i, 0)
        if i < 1
            println("$w_i not in vocabulary")
            continue
        end
        x_i = reshape(embeddings[:,i], size(embeddings, 1), 1)
        D = pairwise(CosineDist(), x_i, embeddings)
        D[i] = Inf
        j = indmin(D)
        d = D[j]
        w_j = idx2word[j]
        println("nn => $w_j")
        println("d => $d\n")
    end
end


length(ARGS) != 1 && throw(ArgumentError("usage => julia word2vec.jl {train|interact}"))
const cmd = ARGS[1]
if cmd == "train"
    train()
elseif cmd == "interact" 
    interact()
else
    throw(ArgumentError("unknown command => $cmd"))
end