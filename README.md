# thrust_dev
thrust api usage, customized api to outperform thrust

### reduce_by_key
input size 896,   key size 124

| GPU      | Thrust Runtime (ms) | Customized Version Runtime (ms) | Seedup |
|----------|---------------------|---------------------------------|--------|
| GTX 950  | 0.033                | 0.018                            | 1.83x  |
| RTX 2070 | 0.037                | 0.014                            | 2.64x |

input size 2047,   key size 124
| GPU      | Thrust Runtime (ms) | Customized Version Runtime (ms) | Seedup |
|----------|---------------------|---------------------------------|--------|
| GTX 950  | 0.033                | 0.022                            | 1.83x  |
