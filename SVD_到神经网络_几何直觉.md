# 从 SVD 到神经网络：用几何直觉理解矩阵、卷积、升维与 ReLU

> 这篇文章从 **Singular Value Decomposition（SVD）** 开始，不重新讲向量、矩阵乘法等基础，而是试图回答一个更有价值的问题：  
> **线性代数里那些“分解、方向、特征值、奇异值”，到底怎样对应到计算机视觉和神经网络内部真正发生的事情？**

---

## 目录

1. [SVD 到底在分解什么](#1-svd-到底在分解什么)
2. [SVD 的三步几何解释](#2-svd-的三步几何解释)
3. [Sigma 的形状：升维和降维到底发生在哪里](#3-sigma-的形状升维和降维到底发生在哪里)
4. [奇异值、rank 与 low-rank approximation](#4-奇异值rank-与-low-rank-approximation)
5. [为什么突然会出现 A^T A](#5-为什么突然会出现-ata)
6. [SVD 和 eigendecomposition 的关系](#6-svd-和-eigendecomposition-的关系)
7. [计算机视觉：为什么 rank-1 filter 是 separable](#7-计算机视觉为什么-rank-1-filter-是-separable)
8. [Fourier / FFT 和 SVD 加速 convolution 有什么区别](#8-fourier--fft-和-svd-加速-convolution-有什么区别)
9. [把神经网络的 linear layer 用 SVD 看一遍](#9-把神经网络的-linear-layer-用-svd-看一遍)
10. [2 维输入为什么要升到 16 维](#10-2-维输入为什么要升到-16-维)
11. [ReLU 到底怎样产生 nonlinearity](#11-relu-到底怎样产生-nonlinearity)
12. [16 个 neuron 为什么能产生超过 16 个线性区域](#12-16-个-neuron-为什么能产生超过-16-个线性区域)
13. [“二维纸折进 16D”这个比喻到底准确到什么程度](#13-二维纸折进-16d这个比喻到底准确到什么程度)
14. [进一步联系：Hessian、symmetry 与 flat directions](#14-进一步联系hessiansymmetry-与-flat-directions)
15. [常见误区](#15-常见误区)
16. [最后的 mental model](#16-最后的-mental-model)

---

# 1. SVD 到底在分解什么

给定任意实矩阵

$$
A\in\mathbb R^{m\times n},
$$

SVD 写成

$$
A=U\Sigma V^T.
$$

在 full SVD 中：

$$
U\in\mathbb R^{m\times m},\qquad
\Sigma\in\mathbb R^{m\times n},\qquad
V\in\mathbb R^{n\times n}.
$$

其中：

- $U$ 的 columns 是 **left singular vectors**；
- $V$ 的 columns 是 **right singular vectors**；
- $U,V$ 都是 orthogonal matrices；
- $\Sigma$ 是矩形 diagonal matrix；
- $\Sigma$ 对角线上的非负数 $\sigma_1,\sigma_2,\ldots$ 是 **singular values**。

所以

$$
U^TU=UU^T=I,\qquad V^TV=VV^T=I.
$$

SVD 最重要的意义不是“把一个矩阵拆成三个矩阵”，而是：

> **找到这个线性变换最自然的一组输入方向和输出方向，让原本复杂的矩阵作用变成几个互不干扰的一维缩放。**

---

# 2. SVD 的三步几何解释

![SVD 三步几何流程](assets/01_svd_flow.svg)

考虑

$$
y=Ax=U\Sigma V^Tx.
$$

可以把它理解成三步。

## 第一步：$V^T$ —— 分析输入

因为 $V$ 的 columns

$$
v_1,\ldots,v_n
$$

是一组 orthonormal directions，

$$
V^Tx=
\begin{bmatrix}
v_1^Tx\\
v_2^Tx\\
\vdots
\end{bmatrix}.
$$

每个 $v_i^Tx$ 都是在问：

> **输入 $x$ 在方向 $v_i$ 上有多少成分？**

如果使用的是完整 $V$，这里更准确地叫 **change of coordinates**，不是降维 projection。  
只有在 truncated / reduced setting 中只保留部分 directions 时，才开始有“投影到低维 subspace”的味道。

## 第二步：$\Sigma$ —— 每个方向独立缩放

如果

$$
V^Tx=c,
$$

那么

$$
\Sigma c
$$

会把第 $i$ 个 coefficient 乘上 $\sigma_i$。

- $\sigma_i>1$：stretch；
- $0<\sigma_i<1$：shrink；
- $\sigma_i=0$：这个方向被完全压没。

## 第三步：$U$ —— 在输出空间重新组合

最后

$$
U(\Sigma V^Tx)
$$

把已经缩放的 coefficients 沿着输出空间的 directions

$$
u_1,u_2,\ldots
$$

重新组合。

因此一个特别值得记住的展开式是：

$$
\boxed{
Ax=\sum_i \sigma_i u_i(v_i^Tx)
}
$$

每一项都可以读成：

> 测输入沿 $v_i$ 有多少  
> $\rightarrow$ 乘上这一 channel 的强度 $\sigma_i$  
> $\rightarrow$ 沿输出方向 $u_i$ 写回。

---

# 3. Sigma 的形状：升维和降维到底发生在哪里

一个常见误区是认为 $\Sigma$ 总是 $n\times n$。

实际上，如果

$$
A\in\mathbb R^{m\times n},
$$

那么 full SVD 中

$$
\Sigma\in\mathbb R^{m\times n}.
$$

整个维度流程是：

$$
\mathbb R^n
\xrightarrow{V^T}
\mathbb R^n
\xrightarrow{\Sigma}
\mathbb R^m
\xrightarrow{U}
\mathbb R^m.
$$

所以：

- $V^T$ 在 input space 内换坐标，不改变维度；
- $U$ 在 output space 内重新组合，不改变维度；
- **真正把 $n$ 维 shape 变成 $m$ 维 shape 的是矩形 $\Sigma$。**

例如 $A\in\mathbb R^{3\times2}$：

$$
\Sigma=
\begin{bmatrix}
\sigma_1&0\\
0&\sigma_2\\
0&0
\end{bmatrix}.
$$

它把

$$
\begin{bmatrix}c_1\\c_2\end{bmatrix}
$$

变成

$$
\begin{bmatrix}
\sigma_1c_1\\
\sigma_2c_2\\
0
\end{bmatrix}.
$$

这里从 2D 进入了 3D ambient space，但**并没有凭空创造第三个独立自由度**。

反过来，如果 $A\in\mathbb R^{2\times3}$，

$$
\Sigma=
\begin{bmatrix}
\sigma_1&0&0\\
0&\sigma_2&0
\end{bmatrix},
$$

第三个 singular-coordinate 无法传到输出，于是会发生真正的压缩。

---

# 4. 奇异值、rank 与 low-rank approximation

SVD 给了一个非常自然的 rank 解释：

$$
\boxed{
\operatorname{rank}(A)
=
\text{non-zero singular values 的数量}
}
$$

但 rank 是非常粗的量。

![Singular values 与 effective rank](assets/04_low_rank.svg)

假设

$$
\sigma_1=100,\qquad
\sigma_2=8,\qquad
\sigma_3=0.0001.
$$

严格来说：

$$
\operatorname{rank}(A)=3.
$$

但第三个方向几乎没有作用。

所以工程上我们可能近似：

$$
A
\approx
\sigma_1u_1v_1^T+
\sigma_2u_2v_2^T.
$$

这就是 **low-rank approximation** 的核心直觉。

因此可以这样区分：

> **rank 告诉你有多少个非零方向；singular values 告诉你这些方向分别有多强。**

现实中的数据和浮点计算经常不会产生“精确等于 0”的 singular value，因此还会出现 numerical rank / effective rank 的概念。

---

# 5. 为什么突然会出现 A^T A

初学 SVD 时，最突兀的一步通常是：

> 为什么讲着讲着突然开始研究 $A^TA$？

如果只看代数，很像是为了把 SVD 里的 $U^TU$ 消掉而硬凑出来。

但它其实有一个自然几何意义。

![A^T A 几何桥梁](assets/02_ata_bridge.svg)

假设

$$
A:\mathbb R^n\rightarrow\mathbb R^m.
$$

我们想问：

> **A 对某个输入方向 $x$ 到底作用多强？**

最自然的量是输出长度：

$$
\|Ax\|_2^2.
$$

展开：

$$
\|Ax\|_2^2
=
(Ax)^T(Ax)
=
x^TA^TAx.
$$

所以 $A^TA$ 天然描述了：

> **A 对 input space 中不同方向的拉伸强度结构。**

而且如果 $A$ 是矩形矩阵，不能直接写一般的 eigen equation

$$
Av=\lambda v
$$

来研究输入方向，因为 $v\in\mathbb R^n$，但 $Av\in\mathbb R^m$，两边甚至不在同一个空间。

而

$$
A^TA:\mathbb R^n\rightarrow\mathbb R^n,
$$

重新回到了 input space，所以可以做 eigendecomposition。

---

# 6. SVD 和 eigendecomposition 的关系

从

$$
A=U\Sigma V^T
$$

得到

$$
A^T=V\Sigma^TU^T.
$$

于是

$$
A^TA
=
V\Sigma^TU^TU\Sigma V^T.
$$

由于

$$
U^TU=I,
$$

所以

$$
\boxed{
A^TA=V\Sigma^T\Sigma V^T
}
$$

这正是 symmetric matrix 的 eigendecomposition 形式。

因此：

$$
\boxed{
A^TA\text{ 的 eigenvectors}
=
A\text{ 的 right singular vectors}
}
$$

并且

$$
\boxed{
\lambda_i(A^TA)=\sigma_i^2
}
$$

所以

$$
\sigma_i=\sqrt{\lambda_i}.
$$

这也给 singular value 一个直接几何解释。

如果 $v_i$ 是单位 right singular vector，

$$
A^TAv_i=\sigma_i^2v_i,
$$

那么

$$
\|Av_i\|_2^2
=
v_i^TA^TAv_i
=
\sigma_i^2,
$$

因此

$$
\boxed{
\|Av_i\|_2=\sigma_i.
}
$$

也就是说：

> $v_i$ 是 A 的特殊输入方向，$\sigma_i$ 就是 A 沿这个方向的 stretch factor。

同理，

$$
AA^T
$$

的 eigenvectors 是 $U$ 的 columns。

---

# 7. 计算机视觉：为什么 rank-1 filter 是 separable

这是 SVD 在计算机视觉里一个非常漂亮的落地。

考虑一个 2D filter/kernel $F$：

$$
F=U\Sigma V^T
=
\sum_i \sigma_i u_iv_i^T.
$$

如果只有一个 singular value 非零：

$$
\sigma_1\neq0,\qquad
\sigma_2=\sigma_3=\cdots=0,
$$

那么

$$
\boxed{
F=\sigma_1u_1v_1^T
}
$$

因此

$$
\operatorname{rank}(F)=1.
$$

这时 $F$ 可以写成两个 1D vectors 的 outer product：

$$
F=ab^T.
$$

![Rank-1 separable filter](assets/03_separable_filter.svg)

例如：

$$
F=
\begin{bmatrix}
1&0&-1\\
2&0&-2\\
1&0&-1
\end{bmatrix}
=
\begin{bmatrix}
1\\2\\1
\end{bmatrix}
\begin{bmatrix}
1&0&-1
\end{bmatrix}.
$$

于是一个 2D convolution 可以分解成：

1. horizontal 1D convolution；
2. vertical 1D convolution。

如果 kernel 是 $k\times k$：

- 普通 2D convolution：每个位置大约 $k^2$ 次乘法；
- separable convolution：大约 $2k$ 次。

但必须注意：

> **不是所有 kernel 都能精确拆成一对 1D filters。**

如果 rank 是 $r$：

$$
F=
\sum_{i=1}^r\sigma_i u_iv_i^T,
$$

它可以表示为 **$r$ 个 separable filters 的和**。

如果后面的 singular values 很小，还可以只保留前几个做 approximate separability。

---

# 8. Fourier / FFT 和 SVD 加速 convolution 有什么区别

SVD / separability 和 Fourier / FFT 都可能加速 convolution，但原理完全不同。

## SVD / separability

思路是：

> **kernel 本身有没有低秩结构？**

如果

$$
F\approx ab^T,
$$

就把 2D filter 拆成两个 1D filters。

## Fourier / FFT

Fourier transform 把信号从 spatial domain 改写到 frequency domain。

卷积定理：

$$
\mathcal F(I*F)
=
\mathcal F(I)\odot\mathcal F(F).
$$

也就是说：

> spatial domain 中昂贵的 convolution  
> 在 frequency domain 中变成 element-wise multiplication。

流程是：

$$
I
\xrightarrow{\text{FFT}}
\hat I,
\qquad
F
\xrightarrow{\text{FFT}}
\hat F,
$$

然后

$$
\hat I\odot\hat F,
$$

最后 inverse FFT。

通常小 kernel（例如 $3\times3$）直接 spatial convolution 很合适；非常大的 kernel 或长信号时 FFT-based convolution 更有吸引力。

不要把两者混在一起：

- SVD：利用 **low-rank structure**；
- FFT：利用 **convolution theorem**。

---

# 9. 把神经网络的 linear layer 用 SVD 看一遍

神经网络最普通的一层：

$$
z=Wx+b.
$$

先忽略 bias：

$$
Wx=U\Sigma V^Tx.
$$

于是一个 weight matrix 可以理解成：

1. $V^T$：分析当前 representation 中的特殊输入 directions；
2. $\Sigma$：选择性放大、压缩甚至删除这些 directions；
3. $U$：把它们重新组合成新的 hidden representation。

也就是：

$$
\boxed{
Wx=\sum_i\sigma_i u_i(v_i^Tx)
}
$$

可以把每一个 $i$ 看成一个 directional channel：

> “输入有多少这种 feature combination？”  
> $\rightarrow$ “这一层对它多敏感？”  
> $\rightarrow$ “把它写到哪个输出 feature direction？”

这里的 $v_i$ 不一定对应肉眼可解释的“横边缘”“竖边缘”。它们首先是高维 representation space 中的 directions。

还要强调：

> **神经网络 forward pass 通常不会真的先计算 SVD。**  
> SVD 是我们理解一个已经学习好的 $W$ 的分析镜头。

---

# 10. 2 维输入为什么要升到 16 维

假设

$$
x\in\mathbb R^2,
$$

第一层有 16 个 neuron：

$$
z=Wx+b,\qquad
W\in\mathbb R^{16\times2}.
$$

那么

$$
z\in\mathbb R^{16}.
$$

如果没有 activation，所有可能的 $z$ 仍然最多只形成一个 2D affine subspace，因为

$$
\operatorname{rank}(W)\le2.
$$

所以：

> **纯 linear 的 2→16 并不会把“2 个自由度”变成“16 个独立自由度”。**

那为什么还要 16 维？

因为 16 个 coordinates 可以成为 16 个不同的 learned feature measurements：

$$
z_i=w_i^Tx+b_i.
$$

每个 neuron 都在对同一个输入问一个不同的问题，例如：

- 沿这个 direction 有多强？
- 是否跨过这个 threshold？
- 这个 feature combination 是否存在？

所以 hidden width 的价值不是“创造更多原始信息”，而是：

$$
\boxed{
\text{提供更多表示同一输入的 learned feature coordinates。}
}
$$

---

# 11. ReLU 到底怎样产生 nonlinearity

加入 ReLU：

$$
h=\operatorname{ReLU}(Wx+b).
$$

对第 $i$ 个 neuron：

$$
h_i(x)
=
\max(0,w_i^Tx+b_i).
$$

在二维输入空间里，

$$
w_i^Tx+b_i=0
$$

是一条直线。

它把输入空间切成两边：

- 一边 neuron active；
- 一边 neuron 输出 0。

如果某个区域内 active / inactive pattern 不变，可以用一个 diagonal mask $D$ 表示：

$$
D=\operatorname{diag}(0/1,\ldots,0/1).
$$

在这个区域内：

$$
\operatorname{ReLU}(W_1x+b_1)
=
D(W_1x+b_1).
$$

如果再接一个线性输出层：

$$
f(x)=W_2\operatorname{ReLU}(W_1x+b_1)+b_2,
$$

那么在该区域：

$$
f(x)
=
W_2DW_1x+W_2Db_1+b_2.
$$

也就是：

$$
\boxed{
f(x)=A_{\text{region}}x+c_{\text{region}}
}
$$

所以：

> **每个小区域内部都是 affine 的，但不同区域使用不同 affine rule。**

这就是 piecewise-linear network 的核心。

---

# 12. 16 个 neuron 为什么能产生超过 16 个线性区域

![ReLU regions](assets/05_relu_regions.svg)

关键是：

> **一个 neuron 不是一个 region，而是一条切割边界 / 一个 gate。**

16 个 neuron 给 16 条二维直线。

这些线的组合可以产生很多不同的 activation patterns。

对于 2D input，$n$ 条一般位置的直线最多可以产生：

$$
\sum_{k=0}^2 {n\choose k}
=
1+n+\frac{n(n-1)}2.
$$

当 $n=16$：

$$
1+16+120=137.
$$

所以 16 个 neuron 最多可以产生 137 个 linear regions（一般位置下）。

每一个 region：

- 对应一组固定的 ReLU on/off pattern；
- 因此对应一套固定的局部 affine transformation。

这里的 region 不是“一个 feature”；feature 是 neuron 输出 $h_i(x)$。  
region 是一类输入：它们触发了相同的一组 gates。

---

# 13. “二维纸折进 16D”这个比喻到底准确到什么程度

![二维纸折进 16D 的概念图](assets/06_folded_2d_in_16d.png)

这是一个非常有用的视觉比喻：

- 输入本质只有 2 个自由度；
- linear 2→16 先把它嵌到 16D ambient space；
- ReLU 根据不同输入区域使用不同 mask；
- 不同区域因此被送到不同方向的 affine patches。

所以：

> **局部仍然最多是二维的，但整体不再是一个单一二维 affine plane。**

可以把它想成一张二维纸被折成很多平面片，然后放进高维空间。

但必须加两个 caveat：

1. ReLU 可能把某些方向压成 0，因此局部维度也可能下降；
2. 不同区域可能重叠、自交或发生信息丢失，所以任意网络的 image 不一定严格是漂亮光滑的 2D manifold。

因此这张图是 **几何直觉**，不是所有 ReLU 网络的严格拓扑描述。

---

# 14. 进一步联系：Hessian、symmetry 与 flat directions

SVD / eigendecomposition 提供了一种很通用的语言：

> **在高维问题里找天然方向，再看每个方向的强度。**

这个思路不仅能分析 weight matrix，也能分析 optimization。

在 minimum 附近，loss 的二阶结构由 Hessian：

$$
H=\nabla^2L(\theta)
$$

描述。

如果

$$
Hv_i=\lambda_i v_i,
$$

那么：

- $v_i$：parameter space 中一个特殊方向；
- 大 $\lambda_i$：沿这个方向稍微移动，loss 上升很快，比较 sharp；
- 小 $\lambda_i$：沿这个方向移动，loss 变化很慢，比较 flat；
- $\lambda_i\approx0$：可能存在 redundancy、symmetry 或非常平的方向。

神经网络确实存在很多 parameter symmetries。

例如隐藏 neuron permutation：

> 交换两个 hidden neurons，同时正确交换后续连接，网络函数不变。

ReLU 还有 positive scaling symmetry：

$$
a\,\operatorname{ReLU}(w^Tx)
=
\frac{a}{c}\operatorname{ReLU}(cw^Tx),\qquad c>0.
$$

所以不同参数点可以代表完全相同的函数。

这会让 loss landscape 出现等价解和 flat directions。

但不要过度总结成：

> “神经网络容易训练就是因为对称性导致最优解很多。”

优化成功还和 overparameterization、gradient dynamics、normalization、architecture、data structure 等很多因素有关。Symmetry 只是其中一部分。

---

# 15. 常见误区

## 误区 1：$V^T$ 就是在降维 projection

不一定。

完整 SVD 中 $V^T$ 是 orthogonal coordinate change，不丢信息。

只有只保留部分 singular directions 时，才进入 truncated projection / compression。

---

## 误区 2：$\Sigma$ 一定是 $n\times n$

不对。

如果

$$
A\in\mathbb R^{m\times n},
$$

则 full SVD 的

$$
\Sigma\in\mathbb R^{m\times n}.
$$

---

## 误区 3：升到 16D 就创造了 16 个独立信息维度

不对。

如果输入只有 2 个自由度，纯 linear map 之后仍然最多只有 2 个独立自由度。

高维 hidden space 的价值主要是 **更丰富的 feature coordinates**。

---

## 误区 4：ReLU 创造新信息

不对。

如果两个输入在 ReLU 前已经变成完全相同的 representation，后面的 deterministic activation 无法把原信息恢复出来。

ReLU 提供的是 **nonlinear expressive power**，不是凭空增加 information。

---

## 误区 5：ReLU 把负半边“从空间里删掉”

更准确地说，它把某个 neuron 在负区域的输出压成 0。

输入点仍然存在；而且别的 neurons 可能从其他方向保留信息。

---

## 误区 6：一个 separable kernel 的条件只是“singular values 很小”

精确 separable 要求：

$$
\operatorname{rank}(F)=1,
$$

也就是只有一个 singular value 非零。

如果后面的 singular values 很小，只能说 **近似 separable / low-rank approximable**。

---

# 16. 最后的 mental model

如果只保留一套脑内图景，可以是：

## SVD

$$
A=U\Sigma V^T
$$

等价于：

> **输入方向分析 → 每个方向独立缩放 → 输出方向重新组合**

或者：

$$
Ax
=
\sum_i \sigma_i u_i(v_i^Tx).
$$

---

## Singular values

> rank 告诉“有多少有效方向”；  
> singular values 告诉“每个方向有多强”。

---

## $A^TA$

> 把 A 的作用拉回 input space，研究 A 对不同输入方向的 stretch structure。

$$
A^TA
=
V\Sigma^T\Sigma V^T.
$$

所以：

$$
\lambda_i(A^TA)=\sigma_i^2.
$$

---

## Computer Vision filter

如果只有一个 singular value 非零：

$$
F=\sigma_1u_1v_1^T,
$$

则 filter rank 1、separable，可以拆成两个 1D convolutions。

---

## Neural network linear layer

$$
Wx
$$

是在：

> **重新组织 representation 的方向和强度。**

---

## Hidden width

2→16 不是“创造 14 个新自由度”，而是：

> **为同一个输入构造 16 个 learned feature coordinates。**

---

## ReLU

不是简单“再做一次变换”，而是：

> **根据输入位置改变 active linear rule。**

因此整个网络：

> 每个局部区域内部是 affine；  
> 很多区域拼起来后形成复杂 nonlinear function。

---

## 一句话总结

> **线性代数里的 matrix decomposition，本质是在寻找更聪明的坐标系，让复杂的高维 transformation 暴露出少数重要方向；而神经网络则在这些线性 transformation 之间插入 nonlinearity，使不同输入区域可以被不同方式重新编码。**

---

## 参考资料与说明

这篇文章的线性代数核心基于 *Deep Learning*（Goodfellow, Bengio, Courville）Chapter 2 中关于 orthogonal matrices、eigendecomposition、SVD、pseudoinverse 和 PCA 的内容进行整理；神经网络、ReLU piecewise-affine geometry、hidden width、卷积 separability、FFT 与 Hessian/symmetry 部分用于建立机器学习与计算机视觉中的直觉联系。

