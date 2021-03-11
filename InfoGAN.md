# InfoGAN

The process is very manual: 1) generate a bunch of images, 2) find images that have the characteristic you want, 3) average together their noise vectors and hope that it captures the structure of interest.

InfoGAN tries to solve this problem and provide a disentangled representation. The way InfoGAN approaches this problem is by splitting the Generator input into two parts: the traditional noise vector and a new “latent code” vector.

## Loss

Mutual Information 개념. Lower Bound에 estimate해서, Lower Bound를 최대화.

“auxiliary” distribution Q(c|x), which is modeled by a parameterized neural network, and is meant to approximate the real P(c|x).

![loss](https://miro.medium.com/max/686/1*NTYmbgNBT9RzhdLl71-koA.png)

Sample a value for the latent code c from a prior of your choice; Sample a value for the noise z from a prior of your choice; Generate x = G(c,z); Calculate Q(c|x=G(c,z)).

## Architecture

![ar](https://miro.medium.com/max/875/1*dXLgTV8lNiTInvxomgZSAg.png)

Discriminator에 Q 네트워크가 추가됨. (Likelihood 계산. C를 직접 계산할 필요 없음.)

The auxiliary distribution introduced in the theory section is modeled by another neural network, which really is just a fully connected layer tacked onto the last representation layer of the discriminator. The Q network is essentially trying to predict what the code is (see nuance below). This is only used when feeding in fake input, since that’s the only time the code is known.

you need to estimate the likelihood of seeing that code for the given generated input. Therefore, the output of Q is not the code value itself, but instead the statistics of the distribution you chose to model the code. Once you know the sufficient statistics of the probability distribution, you can calculate the likelihood.

For instance, if you have used a continuous valued code (i.e. between -1 and +1), you might model Q(c|x) as a Normal/Gaussian distribution. In that case, Q would output two values for this part of the code: the mean and standard deviation. Once you know the mean and standard deviation you can calculate the likelihood Q(c|x), which is what you need for the regularization term.

Categorical일 경우 Cross Entropy 구해주면 됨.

## (Categorical 예시.)

### Discriminator

Q 네트워크는 Discriminator와 상당 부분 (마지막 FC Layer 만 추가) 공유하기 때문에 Computationally 큰 차이가 없다 (약 0.5%)

```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        
        ## Real vs Fake
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))
        
        ## Code (Discrete)
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.n_classes), nn.Softmax())

        ## Code (Continuous) 
        self.latent_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.code_dim))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        return validity, label, latent_code
```


### Generator

단순한 Upsampling 사용.

```python
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        input_dim = opt.latent_dim + opt.n_classes + opt.code_dim

        self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
```

## 학습

(Labeled Data를 사용 -> Discrete variable.)
(MNIST)

뿐만 아니라, continuous variable도 포함. 이는 Generator에서 이미지를 생성할 때 적용. (-1, 0) 사이에서 랜덤, (0, 1) 사이에서 랜덤한 변수를 생성하여 넣어준 결과를 다르게 구분하는 형태.

-> 이 과정을 거치면 Generator input에서 Continuous Variable에 차이에 따라 다른 이미지를 생성하도록 Generator가 학습 될 것이다. 생성된 이미지를 바탕으로 Discriminator가 분포를 추정할 수 있어야 하기 때문이다. Loss는 단순히 MSE 사용해도 되고 `Gaussian Log Likelihood`를 사용하는 것도 방법이 될 수 있다.

~~물론 class variable과는 다르게, continuous variable은 우리가 원하는 효과를 disentangle 할 수는 없을 것이다. (알려 주지 않았으므로...) 하지만 무언가 차이가 생기며, 이는 눈에 띄는 차이라는 것이 핵심이다. 눈에 띄어야지 Discriminator에서 feature를 찾아낼 수 있기 때문이다.~~

```python

# 이미지를 만든다. Generator를 거치는데, z는 random normal, label은 class로 정해지고, c1, c2를 다르게 해서 continuous variable을 조정한다.

def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Static sample
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    static_sample = generator(z, static_label, static_code)
    save_image(static_sample.data, "images/static/%d.png" % batches_done, nrow=n_row, normalize=True)

    # Get varied c1 and c2
    zeros = np.zeros((n_row ** 2, 1))
    c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
    c1 = Variable(FloatTensor(np.concatenate((c_varied, zeros), -1)))
    c2 = Variable(FloatTensor(np.concatenate((zeros, c_varied), -1)))
    sample1 = generator(static_z, static_label, c1)
    sample2 = generator(static_z, static_label, c2)
    save_image(sample1.data, "images/varying_c1/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(sample2.data, "images/varying_c2/%d.png" % batches_done, nrow=n_row, normalize=True)


for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = to_categorical(labels.numpy(), num_columns=opt.n_classes)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        label_input = to_categorical(np.random.randint(0, opt.n_classes, batch_size), num_columns=opt.n_classes)
        code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

        # Generate a batch of images
        gen_imgs = generator(z, label_input, code_input)

        # Loss measures generator's ability to fool the discriminator

        # (generator에서는 validity만 이용하여 학습한다.)
        validity, _, _ = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, _, _ = discriminator(real_imgs)
        d_real_loss = adversarial_loss(real_pred, valid)

        # Loss for fake images
        fake_pred, _, _ = discriminator(gen_imgs.detach())
        d_fake_loss = adversarial_loss(fake_pred, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # ------------------
        # Information Loss
        # ------------------

        # Regulizer를 학습시킨다.

        
        optimizer_info.zero_grad()

        # Sample labels
        sampled_labels = np.random.randint(0, opt.n_classes, batch_size)

        # Ground truth labels
        gt_labels = Variable(LongTensor(sampled_labels), requires_grad=False)

        # Sample noise, labels and code as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        label_input = to_categorical(sampled_labels, num_columns=opt.n_classes)
        code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

        gen_imgs = generator(z, label_input, code_input)
        _, pred_label, pred_code = discriminator(gen_imgs)

        info_loss = lambda_cat * categorical_loss(pred_label, gt_labels) + lambda_con * continuous_loss(
            pred_code, code_input
        )

        info_loss.backward()
        optimizer_info.step()

        # --------------
        # Log Progress
        # --------------

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), info_loss.item())
        )
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)
```