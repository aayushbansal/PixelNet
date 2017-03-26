# Linear with normalization

Without properly normalizing the outputs, we were observing degenerate outputs. We found that [batch-normalization](https://arxiv.org/abs/1502.03167) as an important tool along with our simple technique of _sampling_ to train linear models. The performance of our linear model (with normalization) is similar to one obtained by [Hypercolumn](https://arxiv.org/abs/1411.5752) and [FCN](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf).

Refer to Section 4.2 (Linear vs. MLP) for this folder. 
