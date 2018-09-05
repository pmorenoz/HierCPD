# Hierarchical Change-Point Detection

This repository contains the implementation of our Hierarchical Change-Point Detection (HierCPD) model that is fully written in Python. This model addresses the problem of change-point detection on sequences of **high-dimensional** and **heterogeneous** observations (i.e. different statistical data types) with an unknown temporal structure.

Please, if you use this code, cite the following paper:
```
@article{MorenoRamirezArtes18,
  title={Change-Point Detection on Hierarchical Circadian Models},
  author={Pablo Moreno-Mu\~noz, David Ram\'irez and Antonio Ar\'ets-Rodr\'iguez},
  journal={arXiv preprint arXiv:XXXX.XXXX},
  year={2018}
}
```

The repository is divided in two sections that correspond to the two principal contributions of our work: **(i) Hierarchical Detector** and **(ii) Heterogeneous Circadian Mixture Models**.

# Hierarchical (Bayesian) Change-Point Detector
This is a novel probabilistic extension of the widely known Bayesian Online Change-Point Detection (BOCPD) algorithm [link](http://hips.seas.harvard.edu/content/bayesian-online-changepoint-detection). We extend the method to handle any type of latent variable model. In particular, it is able to detect by directly modeling complex observations from their latent representation embedded in a lower dimensional manifold.

There is a notebook demo of the detector performance.

**Graphical model:** Representation of the hierarchical change-point detector. The detection is performed exclusively on the blue region. Additionally, the green region corresponds to the embedded circadian model (see below).
![graphical_model](tmp/graphical_model.png)


# Heterogeneous Circadian Mixture Models

[Paramz](https://github.com/sods/paramz)

**Circadian Model Infographic:** .

![infographic](tmp/infographic.png)
