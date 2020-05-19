# thrust_dev
thrust api usage, customized api to outperform thrust

### reduce_by_key
input size 896,   key size 124

| GPU      | Thrust Runtime (ms) | Customized Version Runtime (ms) | Seedup |
|----------|---------------------|---------------------------------|--------|
| GTX 950  | 0.33                | 0.18                            | 1.83X  |
| RTX 2070 |                     |                                 |        |
