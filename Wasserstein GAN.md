# Wasserstein GAN




- 박태준, 신준호
- 공간통계연구실
- 2022년 7월 5일



## Topic
![STDNNK](./img/topic.png)



## GAN의 문제점

- Minimizing the GAN objective function with an optimal discriminator, $D^*$, is equivalent to minimizing the JS-divergence;

$$\min_G V(D^*, G) = 2JS(\mathbb p_r, \mathbb p_\theta) - 2\log2$$

- **Claim:** Divergence, which is not continuous with the generator's parameter, leads to difficulty in learning.


### Training GANs is hard for theoretical reasons with the GAN cost functions.
> Arjovsky et al., (2017), Towards principled methods for training Generative Adversarial Networks.






