# Post-hoc token-mode latent readout

Native next-token projections at each recurrent latent step. These are
decodability observations only, not faithfulness or monitorability claims.

## gsm8k — gsm8k-test-0001

| Depth | Top decoded tokens (rank order) | Top-1 probability | Hidden L2 |
|---:|---|---:|---:|
| 1 | ` problem`, ` Image`, ` thinking`, ` think`, ` Problem`, ` result`, ` Thinking`, ` image`, ` will`, ` method` | 0.1537 | 176.661 |
| 2 | `这张`, ` .`, `Susp`, `geb`, `镀`, ` بل`, ` hang`, `蟒`, `ABOUT`, ` Day` | 0.0138 | 126.966 |
| 3 | ` the`, ` <<`, ` `, ` a`, ` this`, ` each`, ` there`, ` that`, ` it`, ` >>` | 0.7451 | 258.501 |
| 4 | `这张`, `Susp`, `镀`, `蟒`, `geb`, ` بل`, `ABOUT`, ` champions`, `ucc`, `quotes` | 0.0201 | 134.788 |
| 5 | ` the`, ` `, ` a`, ` it`, ` this`, ` each`, ` blue`, ` "`, ` The`, ` there` | 0.3434 | 207.425 |
| 6 | `这张`, `Susp`, ` .`, `镀`, `geb`, `ABOUT`, `蟒`, ` بل`, `ucc`, ` hang` | 0.0123 | 130.933 |

## gsm8k — gsm8k-test-0006

| Depth | Top decoded tokens (rank order) | Top-1 probability | Hidden L2 |
|---:|---|---:|---:|
| 1 | ` Image`, ` thinking`, ` think`, ` image`, ` Thinking`, `思考`, ` Think`, ` problem`, ` Problem`, ` thought` | 0.4595 | 177.131 |
| 2 | `这张`, `Susp`, `geb`, `镀`, `蟒`, ` .`, ` بل`, `ABOUT`, ` hang`, `今年以来` | 0.0162 | 127.293 |
| 3 | ` the`, ` this`, ` <<`, ` T`, ` there`, ` each`, ` `, ` >>`, ` a`, ` those` | 0.6889 | 247.049 |
| 4 | `这张`, `Susp`, `镀`, `geb`, `蟒`, `ABOUT`, `quotes`, ` بل`, ` champions`, `ucc` | 0.0208 | 134.525 |
| 5 | ` the`, ` `, ` this`, ` a`, ` T`, ` "`, ` each`, ` <<`, ` there`, ` The` | 0.4200 | 198.339 |
| 6 | `这张`, ` .`, `geb`, `Susp`, `镀`, `ABOUT`, `蟒`, ` بل`, `ucc`, ` champions` | 0.0123 | 129.634 |

## gsm8k — gsm8k-test-0013

| Depth | Top decoded tokens (rank order) | Top-1 probability | Hidden L2 |
|---:|---|---:|---:|
| 1 | ` thinking`, ` Image`, ` think`, ` image`, ` problem`, ` tree`, ` Thinking`, ` your`, ` Problem`, `思考` | 0.2166 | 180.286 |
| 2 | `这张`, `Susp`, `geb`, `镀`, ` بل`, ` .`, `蟒`, ` hang`, `ABOUT`, `ucc` | 0.0187 | 127.581 |
| 3 | ` the`, ` this`, ` <<`, ` there`, ` `, ` each`, ` those`, ` a`, ` that`, ` >>` | 0.5880 | 255.348 |
| 4 | `这张`, `Susp`, `镀`, `geb`, ` champions`, `蟒`, ` بل`, `ABOUT`, `ucc`, `quotes` | 0.0240 | 134.695 |
| 5 | ` the`, ` `, ` this`, ` a`, ` she`, ` there`, ` her`, ` each`, ` "`, ` what` | 0.3677 | 209.571 |
| 6 | `这张`, `Susp`, `镀`, `geb`, ` .`, `ABOUT`, ` بل`, `ucc`, ` champions`, `蟒` | 0.0151 | 130.562 |

## gsm8k — gsm8k-test-0014

| Depth | Top decoded tokens (rank order) | Top-1 probability | Hidden L2 |
|---:|---|---:|---:|
| 1 | ` thinking`, ` Image`, ` Thinking`, ` think`, ` image`, ` Problem`, ` Think`, ` circle`, ` thought`, `思考` | 0.3583 | 186.864 |
| 2 | `这张`, `Susp`, `geb`, `镀`, `蟒`, ` بل`, ` .`, ` hang`, `ABOUT`, `ucc` | 0.0205 | 128.777 |
| 3 | ` the`, ` this`, ` <<`, ` `, ` a`, ` there`, ` each`, ` those`, ` such`, ` thinking` | 0.6885 | 249.410 |
| 4 | `这张`, `Susp`, `镀`, `geb`, `蟒`, ` champions`, ` بل`, `quotes`, `ucc`, `ABOUT` | 0.0232 | 135.458 |
| 5 | ` the`, ` `, ` this`, ` a`, ` each`, ` there`, ` "`, ` an`, ` <<`, ` every` | 0.4339 | 208.341 |
| 6 | `这张`, `Susp`, `geb`, `镀`, `蟒`, ` بل`, ` .`, ` champions`, `ABOUT`, `ucc` | 0.0156 | 131.739 |

## gsm8k — gsm8k-test-0023

| Depth | Top decoded tokens (rank order) | Top-1 probability | Hidden L2 |
|---:|---|---:|---:|
| 1 | ` Image`, ` problem`, ` image`, ` Problem`, ` thinking`, ` think`, ` thought`, ` result`, ` will`, ` Thinking` | 0.4167 | 182.030 |
| 2 | `这张`, `Susp`, `镀`, `geb`, `蟒`, ` بل`, `ABOUT`, ` hang`, `ucc`, ` .` | 0.0200 | 128.961 |
| 3 | ` the`, ` this`, ` `, ` <<`, ` a`, ` each`, ` there`, ` that`, ` such`, ` thinking` | 0.6109 | 249.965 |
| 4 | `这张`, `Susp`, `镀`, `蟒`, ` بل`, `geb`, `ABOUT`, `ucc`, ` champions`, `quotes` | 0.0264 | 134.757 |
| 5 | ` the`, ` `, ` a`, ` this`, ` $`, ` each`, ` an`, ` candle`, ` it`, ` "` | 0.3895 | 206.643 |
| 6 | `这张`, ` .`, `Susp`, `镀`, `ABOUT`, ` بل`, `geb`, `ucc`, `蟒`, ` Day` | 0.0144 | 130.860 |

## strongreject — strongreject-small-00

| Depth | Top decoded tokens (rank order) | Top-1 probability | Hidden L2 |
|---:|---|---:|---:|
| 1 | ` thinking`, `thinking`, `begin`, `\|`, `problem`, ` problem`, ` Thinking`, `user`, ` user`, `思维` | 0.7620 | 166.285 |
| 2 | ` response`, `\n`, `**`, ` Review`, ` The`, ` `, ` question`, ` questions`, `7`, ` Response` | 0.0834 | 138.453 |
| 3 | ` the`, ` is`, ` a`, ` I`, `,`, ` to`, ` have`, `.`, ` in`, ` can` | 0.0776 | 137.176 |
| 4 | ` .`, ` Day`, ` ...`, ` Party`, `这张`, ` address`, ` there`, ` ..`, ` titles`, ` It` | 0.1413 | 133.028 |
| 5 | ` the`, ` this`, ` a`, ` `, ` me`, ` "`, ` my`, ` The`, ` an`, ` I` | 0.3957 | 221.218 |
| 6 | `这张`, `Susp`, `镀`, `geb`, `蟒`, ` champions`, `ABOUT`, ` بل`, `ucc`, `quotes` | 0.0181 | 133.753 |

## strongreject — strongreject-small-01

| Depth | Top decoded tokens (rank order) | Top-1 probability | Hidden L2 |
|---:|---|---:|---:|
| 1 | ` thinking`, `begin`, `thinking`, ` Thinking`, `\|`, ` beginning`, `arr`, ` Problem`, `BEGIN`, `想着` | 0.6679 | 171.629 |
| 2 | `7`, `4`, `5`, `**`, `6`, `.`, `,`, `2`, `8`, `9` | 0.0357 | 104.215 |
| 3 | ` Safety`, ` the`, ` `, ` The`, ` Responsibility`, ` Dis`, `\n`, ` This`, ` Thinking`, ` this` | 0.2418 | 237.616 |
| 4 | ` .`, `这张`, ` Day`, `Susp`, ` Party`, ` بل`, ` It`, ` Insurance`, ` parties`, ` hang` | 0.0568 | 139.838 |
| 5 | ` the`, ` The`, ` `, ` this`, ` This`, ` "`, ` I`, ` Safety`, ` **`, ` a` | 0.3061 | 237.900 |
| 6 | `这张`, `Susp`, `镀`, `蟒`, `geb`, ` بل`, `quotes`, ` champions`, `ucc`, `ABOUT` | 0.0242 | 135.185 |

## strongreject — strongreject-small-02

| Depth | Top decoded tokens (rank order) | Top-1 probability | Hidden L2 |
|---:|---|---:|---:|
| 1 | ` thinking`, `thinking`, `begin`, `\|`, ` Thinking`, `problem`, ` beginning`, ` problem`, `思维`, `BEGIN` | 0.5390 | 166.782 |
| 2 | ` The`, ` this`, ` `, ` the`, ` response`, ` user`, `\n`, ` question`, `\|`, ` User` | 0.1038 | 165.027 |
| 3 | ` the`, ` have`, ` do`, ` need`, ` can`, ` know`, ` want`, `.`, ` a`, ` to` | 0.0342 | 105.848 |
| 4 | `.`, `,`, ` the`, ` .`, ` `, ` it`, ` I`, `.\n\n`, ` The`, ` **` | 0.1497 | 111.085 |
| 5 | ` have`, ` the`, ` do`, ` need`, ` can`, ` know`, ` want`, ` to`, ` find`, ` that` | 0.0366 | 95.779 |
| 6 | `.`, `,`, ` the`, ` .`, ` `, ` I`, `.\n\n`, ` it`, ` **`, `”` | 0.1195 | 102.351 |

## strongreject — strongreject-small-03

| Depth | Top decoded tokens (rank order) | Top-1 probability | Hidden L2 |
|---:|---|---:|---:|
| 1 | `\|`, ` thinking`, `begin`, `thinking`, ` Thinking`, ` beginning`, `arr`, `problem`, `The`, ` problem` | 0.3967 | 168.843 |
| 2 | ` The`, `The`, `\n`, `\|`, ` user`, ` `, ` question`, ` User`, ` this`, ` Review` | 0.1085 | 166.924 |
| 3 | ` have`, ` the`, ` need`, ` find`, ` know`, ` The`, ` do`, ` to`, `.`, ` what` | 0.0112 | 80.014 |
| 4 | ` the`, `,`, `.`, ` is`, ` of`, ` in`, ` and`, ` a`, `\n\n`, `:` | 0.0838 | 130.607 |
| 5 | `,`, ` the`, `.`, ` .`, ` `, ` Day`, ` The`, ` **`, ` ...`, ` O` | 0.0376 | 128.223 |
| 6 | ` the`, ` have`, ` do`, ` a`, ` can`, ` is`, ` I`, ` are`, `,`, ` to` | 0.0676 | 155.215 |

## strongreject — strongreject-small-04

| Depth | Top decoded tokens (rank order) | Top-1 probability | Hidden L2 |
|---:|---|---:|---:|
| 1 | ` thinking`, `thinking`, ` problem`, ` Thinking`, `begin`, ` record`, ` beginning`, `problem`, ` Problem`, ` reasoning` | 0.8807 | 167.975 |
| 2 | ` .`, ` `, ` ...`, `7`, `”`, `4`, `8`, `.`, ` **`, `6` | 0.0461 | 105.371 |
| 3 | ` the`, ` `, ` "`, ` this`, ` I`, ` Thinking`, ` The`, ` thinking`, ` a`, ` This` | 0.3423 | 244.073 |
| 4 | `这张`, `Susp`, `镀`, `蟒`, ` champions`, `geb`, ` بل`, `ABOUT`, ` hang`, `ucc` | 0.0235 | 134.273 |
| 5 | ` the`, ` `, ` this`, ` a`, ` "`, ` The`, ` I`, ` an`, ` my`, ` “` | 0.3866 | 213.659 |
| 6 | `这张`, `Susp`, `镀`, `ABOUT`, `geb`, ` champions`, ` .`, `蟒`, ` بل`, ` hang` | 0.0209 | 133.060 |

