# PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION (by NVIDIA)

[link](https://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of/karras2018iclr-paper.pdf)

The generation of high-resolution images is difficult because higher resolution makes it easier to tell the generated images apart from training images (Odena et al., 2017), thus drastically amplifying the gradient problem. Large resolutions also necessitate using smaller minibatches due to memory constraints, further compromising training stability. Our key insight is that we can `grow both the generator and discriminator progressively`, starting from easier low-resolution images, and add new layers that introduce higher-resolution details as the training progresses. This greatly speeds up training and improves stability in high resolutions

![p](progressive.png)

This incremental nature allows the training to first discover large-scale structure of the image distribution and then `shift attention to increasingly finer scale detail`, instead of having to learn all scales simultaneously.

Another benefit is the reduced training time. With progressively growing GANs most of the iterations are done at lower resolutions, and comparable result quality is often obtained up to 2–6 times faster, depending on the final output resolution.

![p](progressive_smooth.png)

This avoids sudden shocks to the already well-trained smaller-resolution layers.

## INCREASING VARIATION USING MINIBATCH STANDARD DEVIATION

GANs have a tendency to capture only a subset of the variation found in training data, and Salimans et al. (2016) suggest `“minibatch discrimination”` as a solution. **They compute feature statistics not only from individual images but also across the minibatch, thus encouraging the minibatches of generated and training images to show similar statistics. (If not, Discriminator가 fake라고 판별할 것임)** This is implemented by adding a minibatch layer towards the end of the discriminator, where the layer learns a large tensor that projects the input activation to an array of statistics.

We first compute the standard deviation for each feature in each spatial location over the minibatch. We then average these estimates over all features and spatial locations to arrive at a single value. We replicate the value and concatenate it to all spatial locations and over the minibatch, yielding one additional (constant) feature map. This layer could be inserted anywhere in the discriminator, but we have found it best to insert it towards the end.
`(512 x 4 x 4 -> "513" x 4 x 4)`

## NORMALIZATION IN GENERATOR AND DISCRIMINATOR

GANs are prone to the escalation of signal magnitudes as a result of unhealthy competition between the two networks. Most if not all earlier solutions discourage this by using a variant of batch normalization.

We deviate from the current trend of careful weight initialization, and instead use a trivial N (0, 1) initialization and then explicitly scale the weights at runtime. To be precise, we set wˆi = wi/c, where wi are the weights and c is the per-layer normalization constant from He’s initializer (He et al., 2015) `(c = (2/n)^(0.5))`