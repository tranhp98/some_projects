# Project overview
In this project, we would like to check some common properties of the loss functions during training deep learning tasks. In theory, we usually assume the loss to be convex and smooth but that might not be the case due to deep neural networks. For simplicity, let us denote our model parameter as $x,y \in R^d$ and our loss function $f(x): R^d \mapsto R$. Then we want to check the following.
+ Convexity_gap: We compute the additive convexity gap in every iterate as $f(x_t) -f(y) - \langle \nabla f(x_t), x_t-y  \rangle$ where $x_t$ is the current iterate and $y$ is some reference point. We then report the average of this quantity in every epoch (a negative convexity gap means the function is convex).
+ Smoothness:  We compute the smoothness constant $L= \|\nabla f(y) -\nabla f(x_t)\|/\| \|y -x_t\|$ where $x_t$ is the current iterate and $y$ is some reference point. We then report the maximum L of every epoch. 
+ Ratio: We also compute the multiplicative convexity gap which is $\langle \nabla f(x_t), x_t-y  \rangle/(f(x_t) -f(y)) $. We then report the sum of the numerator/sum of the denominator in each epoch (our function is "well-behaved" if this ratio is a positive constant).
+ Exponential average of smoothness and convexity gap: We also compute the exponential average of these quantities to have a better understanding of the local/instantaneous vs the global/average properties.
+ Furthermore, we also want to compute other properties such as the norm of the gradients, the variance of the gradients, $\langle g_t, g_{t-1}  \rangle$,...