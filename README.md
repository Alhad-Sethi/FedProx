### FedProx: Implementation and Evaluation

We present an implementation of the celebrated FedProx algorithm as introduced in [[1]](https://arxiv.org/abs/1812.06127) to tackle client heterogeneity in federated networks. The pseudocode is as follows:

#### **Algorithm:** *FedProx*

**Input:** K, T, $\mu$, $\gamma$, $w^0$, N, $p_k$, $k = 1$, $\cdots$, N  
**for** $t = 0, \cdots, T-1$ **do**  
&nbsp;&nbsp;&nbsp;&nbsp;Server selects a subset  $S_t$ of K devices at random (each device $k$ is chosen with probability $p_k$ )  
&nbsp;&nbsp;&nbsp;&nbsp;Server sends $w^t$ to all chosen devices  
&nbsp;&nbsp;&nbsp;&nbsp;Each chosen device $k \in S_t$ finds a $w_k^{t+1}$ which is an inexact minimizer of:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$w_k^{t+1} \approx \arg \min_w h_k(w; w^t) = F_k(w) + \frac{\mu}{2} \|| w - w^t \||^2$$  
&nbsp;&nbsp;&nbsp;&nbsp;Each device $k \in S_t$ sends $w_k^{t+1}$ back to the server  
&nbsp;&nbsp;&nbsp;&nbsp;Server aggregates the $w$'s as:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$w^{t+1} = \frac{1}{K} \sum_{k \in S_t} w_k^{t+1}$$  
**end for**

In essence, the addition of the $l2$-penalty term in the client losses over the standard FedAvg algorithm [[2]](https://arxiv.org/abs/1602.05629) drives the view and the actual parameter closer together, making for faster convergence and lower client drift.

Further, we perform experiments on i.i.d. (sampling from MNIST) and non-i.i.d. data (sampling from MNIST, fixing labels for each class) and visualize the effect of the the regularization coefficient $\mu$.

---- 
**To reproduce the results:**
Clone the repo and run the following command:
> python3 fedprox.py

----
**TODO:**
- Model partial participation by choosing random subset of clients.
- Model system heterogeneity / stragglers.


----
**Contributors**
- [@abhishek21441](https://github.com/abhishek21441)
- [@AlhadSethi](https://github.com/Alhad-Sethi/)
----
#### References
[1] Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). Federated optimization in heterogeneous networks. Proceedings of Machine learning and systems, 2, 429-450.

[2] McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017, April). Communication-efficient learning of deep networks from decentralized data. In Artificial intelligence and statistics (pp. 1273-1282). PMLR.
