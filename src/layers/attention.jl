struct MHAttention
    heads
    scale
    qkvlayer
    outlayer
end
@functor MHAttention

function MHAttention(planes::Integer, nheads::Integer = 8; qkv_bias::Bool = false,
                     attn_dropout_prob = 0.0, proj_dropout_prob = 0.0)
    # hidden_planes = headplanes * nheads
    # outproject = !(nheads == 1 && headplanes == planes)

    to_qkv = Dense(planes, planes * 3; bias = qkv_bias)
    to_out = Chain(Dense(planes, planes), Dropout(proj_dropout_prob))

    MHAttention(nheads, Float32(sqrt(planes)), to_qkv, to_out)
end

function (m::MHAttention)(x)
    q, k, v = chunk(m.qkvlayer(x), 3; dims = 1)
    @cast q[h, b, d, n] := q[(h, d), n, b] h in 1:m.heads
    @cast k[h, b, d, n] := k[(h, d), n, b] h in 1:m.heads
    @cast v[h, b, d, n] := v[(h, d), n, b] h in 1:m.heads
    dots = NeuralAttentionlib.matmul(q, permutedims(k, (2, 1, 3, 4)), m.scale)
    attn = softmax(dots; dims = 3)
    # PS - this is significantly slower than the one below because `unwrap_collapse` takes a lot of time
    # out = NeuralAttentionlib.unwrap_collapse(NeuralAttentionlib.matmul(attn, v))
    # @cast out[(h, d), n, b] := out[h, b, d, n] in 1:m.heads
    out = NeuralAttentionlib.matmul(attn, v)
    out = reshape(out, :, size(q, 4), size(q, 2))
    m.outlayer(out)
end
