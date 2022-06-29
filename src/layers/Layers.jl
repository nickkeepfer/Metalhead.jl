module Layers

using Flux
using CUDA
using NNlib, NNlibCUDA
using Functors
using ChainRulesCore
using Statistics
using MLUtils
using Random

include("../utilities.jl")

include("attention.jl")
include("embeddings.jl")
include("mlp-linear.jl")
include("normalise.jl")
include("conv.jl")
include("drop.jl")
include("selayers.jl")
include("classifier.jl")

export MHAttention,
       PatchEmbedding, ViPosEmbedding, ClassTokens,
       mlp_block, gated_mlp_block,
       LayerScale, DropPath, DropBlock,
       ChannelLayerNorm, prenorm,
       skip_identity, skip_projection,
       conv_bn, depthwise_sep_conv_bn,
       squeeze_excite, effective_squeeze_excite,
       invertedresidual, create_classifier
end
