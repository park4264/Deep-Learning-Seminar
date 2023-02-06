# Wasserstein GAN




- 박태준, 신준호
- 공간통계연구실
- 2022년 7월 5일



## Topic
![img](./img/topic.png)



## GAN의 문제점

- Minimizing the GAN objective function with an optimal discriminator, $D^*$, is equivalent to minimizing the JS-divergence;

$$\min_G V(D^*, G) = 2JS(\mathbb p_r, \mathbb p_\theta) - 2\log2$$

- **Claim:** Divergence, which is not continuous with the generator's parameter, leads to difficulty in learning.


### Training GANs is hard for theoretical reasons with the GAN cost functions.
> *Arjovsky et al., (2017), Towards principled methods for training Generative Adversarial Networks.*

- When $\mathbb p_r$ and $\mathbb p_\theta$ lie on low dimensional manifolds, there's always a perfect discriminator that can be trained well.
- It provides no usable gradients. ( $\nabla D^*(x)$ will be 0 for almost everywhere.)
  - Gradient vanishing:
    $$\nabla_{\theta_g} \log \Big( 1 - D(G(z^{(i)})) \Big)  \rightarrow 0$$
    under optimal discriminator. ( $D$ is close to $D^*$)
  - Mode collapse:
    $$-\nabla_{\theta_g}\log D(G(z^{(i)}))$$ 
    $$\textit{unstable with large variance of gradients.}$$


### Discriminator vs Critic

![img](./img/vs.png)








- No longer have to worry about the fast learning of the discriminator.
- The gradient is smoother everywhere and learns better even the generator is not producing good images.

## Introduction

- We focus on the ways to measure how close $\mathbb p_\theta$ is to $\mathbb p_r$ ; 
$$\rho(\mathbb p_\theta, \mathbb p_r)$$
- The most fundamental difference:
    Their impact on the convergence of sequence of probability distribution.
- Note that: 
    A sequence of distribution $(\mathbb p_t)_{t\in \mathbb N}$ **converges**
    
    $\Leftrightarrow$ $^\exists \mathbb p_{\infty}$ s.t. $\rho(\mathbb p_t, \mathbb p_{\infty})$ tends to 0.
    
    - We want to find a weaker metric $\rho$.

- In order to optimize the parameter $\theta$ , it is desirable to define our model
distribution $\mathbb p_\theta$ in a manner that makes the mapping $\theta \mapsto \mathbb p_\theta$ continuous.

- **Continuity:** when a sequence of parameters $\theta_t$ converges to $\theta,$ the distribution $\mathbb p_{\theta_t}$ also converge to $\mathbb p_\theta.$

  - It depends on the way we compute the distance between distributions.

- The main reason we care about the mapping $\theta \mapsto \mathbb p_\theta$ to be continuous: 
  - we would like to have a loss function $\theta \mapsto \rho(\mathbb p_\theta, \mathbb p_r)$ that is continuous, and this is equivalent to having the mapping $\theta \mapsto \mathbb p_\theta$ be continuous.

    
> Note that for $f: \{ \theta_\alpha \} \rightarrow \{\mathbb p_\beta \}$,$~f(\theta) = \mathbb p_\theta$, $f(\theta)$ is continuous if
$$~^\forall \text{open } V \subset \{ \mathbb p_\beta \},~ f^{-1}(V) \text{ is also open in } \{\theta_\alpha \}$$
