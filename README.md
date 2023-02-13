# Deep Learning Study

- 서울대학교 공간통계연구실 딥러닝 세미나 (2022.07~2022.09) 에서 다룬 논문들을 리뷰하고 구현해봅니다.
- 수학, 통계학을 공부 후 처음 딥러닝을 공부하고 관심 갖게되는 계기가 되었습니다.
- 딥러닝 세미나의 전체적인 계획과 관련 논문 엄선은 개인적으로 공부에 많은 도움을 준 유호준 박사 (Ph.D.2022, Postdoctoral Fellow in University of Houston, USA)가 진행했습니다.

---

- 세미나에서 제안된 논문은 다음과 같습니다.

1. Kingma and Welling (2013) - Variational Autoencoder [[pdf]](https://arxiv.org/abs/1312.6114)
2. Goodfellow et al. (2014) - Generative Adversarial Nets [[pdf]](https://arxiv.org/abs/1406.2661)
3. Arjovsky et al. (2017) - Wasserstein GAN [[pdf]](https://arxiv.org/abs/1701.07875) [[Review]](https://github.com/park4264/Deep-Learning-Seminar/blob/main/1.%20Wasserstein%20GAN.md) [[Beamer]](https://github.com/park4264/Deep-Learning-Seminar/blob/main/Beamer/WGAN_Beamer.pdf)
4. Chen et al. (2018) - Neural ordinary differential equations [[pdf]](https://arxiv.org/abs/1806.07366)
5. Song et al. (2021) - Score-based generative modeling through stochastic differential equations [[pdf]](https://openreview.net/forum?id=PxTIG12RRHS)
6. Vaswani et al. (2017) - Attention is all you need [[pdf]](https://arxiv.org/abs/1706.03762) 
7. Gal and Ghahramani (2016) - Dropout as a bayesian approximation_representing model uncertainty in deep learning [[pdf]](https://arxiv.org/abs/1506.02142) [[Review]](https://github.com/park4264/Deep-Learning-Seminar/blob/main/2.%20Dropout%20as%20a%20bayesian%20approximation_representing%20model%20uncertainty%20in%20deep%20learning.md) [[Beamer]](https://github.com/park4264/Deep-Learning-Seminar/blob/main/Beamer/MC%20Dropout_Beamer.pdf)
8. Gal and Ghahramani (2016) - Bayesian convolutional neural networks with bernoulli approximate variaional inference [[pdf]](https://arxiv.org/abs/1506.02158) [[Review]]() [[Beamer]](https://github.com/park4264/Deep-Learning-Seminar/blob/main/Beamer/Bayesian%20CNNs_Beamer.pdf)
9. Wu et al. (2020) - A comprehensive survey on graph neural networks [[pdf]](https://arxiv.org/abs/1901.00596) 
10. Kipf and Welling (2017) - Semi-supervised classification with graph convolutional networks [[pdf]](https://arxiv.org/abs/1609.02907)
11. Bellemare et al. (2017) - A distributional perspective on reinforcement learning [[pdf]](https://arxiv.org/abs/1707.06887) 
12. Levine (2018) - Reinforcement learning and control as probabilistic inference_tutorial and review [[pdf]](https://arxiv.org/abs/1805.00909)
13. Finn et al. (2017) - Model-agnostic meta learning for fast adaptation of deep networks [[pdf]](https://arxiv.org/abs/1703.03400)
14. Finn et al. (2018) - Probabilistic model-agnostic meta-learning [[pdf]](https://arxiv.org/abs/1806.02817)
15. He et al. (2020) - Momentum contrast for unsupervised visual representation learning [[pdf]](https://arxiv.org/abs/1911.05722)
16. Chen et al. (2020) - A simple framework for contrastive learning of visual representations [[pdf]](https://arxiv.org/abs/2002.05709)



---

- 이 중 제가 맡아 발표한 아래의 논문에 대한 리뷰를 정리합니다.

  - Arjovsky et al. (2017). Wasserstein GAN
  - Gal and Ghahramani (2016). Dropout as a bayesian approximation_representing model uncertainty in deep learning
  - Gal and Ghahramani (2016). Bayesian convolutional neural networks with bernoulli approximate variaional inference
