| implementation | dtype | seq_len | d_model | forward_ms | backward_ms | end_to_end_ms | status | note |
|---|---|---:|---:|---:|---:|---:|---|---|
| pytorch_attention | bfloat16 | 128 | 16 | 0.02456327339380302 | 0.25466174269333863 | 0.4873205333948135 | ok |  |
| triton_flashattention2 | bfloat16 | 128 | 16 | 0.005246596559963859 | 0.8202879970723932 | 1.1936447938283286 | ok |  |
