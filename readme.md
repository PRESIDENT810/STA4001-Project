# STA4001 Project Report

ZHONG Kaining 117010420

## 1. Environment Required

Interpreter: Python3.7

Library: numpy, pandas, matplotlib, tqdm, torch

OS environment: Mac OS Catalina

## 2. Monte Carlo

In question 1, we are required to use Monte Carlo method to simulate the expected profit for a series of given states. The given states are sampled from a DTMC.



For the original DTMC, at each period there is a state $X_t$: $X_t=(x_{t,1},x_{t,2},...x_{t,L-1})$, where $x_{t,i}\ means\ the\ number\ of\ items\ with\ remaining\ lifetime\ i\ in\ time\ t$


We Need to generate $T$ periods, where $T = 10^3K, K=20,200,2000$


In the end, a array 
$$
    \left[
	\begin{matrix}{}
    x_{1,1} & x_{1,2} & x_{1,3} & \cdots & x_{1,L-1}\\
    x_{2,1} & x_{2,2} & x_{2,3} & \cdots & x_{2,L-1}\\
    \vdots & \vdots & \vdots & \vdots & \vdots\\    
    x_{T,1} & x_{T,2} & x_{T,3} & \cdots & x_{T,L-1}\\
	\end{matrix}
	\right]
$$

should be gerenated, and $x_{i,j}$ means the number of items in time i with j life-period remaining

After this DTMC is generated, we need to sample K states from this DTMC.

$\bar x$ is the sample from DTMC where $\bar x = \{\bar x_1, \bar x_2, \bar x_3, ...\bar x_k \}, \bar x_i = x_{1000*i}$


The result is a matrix:
$$
\left[
	\begin{matrix}{}
    x_{1000,1} & x_{1000,2} & x_{1000,3} & \cdots & x_{1000,L-1}\\
    x_{2000,1} & x_{2000,2} & x_{2000,3} & \cdots & x_{2000,L-1}\\
    \vdots & \vdots & \vdots & \vdots & \vdots\\    
    x_{1000K,1} & x_{1000K,2} & x_{1000K,3} & \cdots & x_{1000K,L-1}\\
	\end{matrix}
	\right]
$$


Then use Monte Carlo method to compute the estimated profit of these states.

Use Monte Carlo method to simulate $$v(x) = E[\sum_{n=0}^\infty \beta^n g(X_n)|X_0=x]$$


To implement Monte Carlo method, first need to calcute a episode: $episode(i)=\sum_{n=0}^L \beta^n g(X_n)$

$g(X_n)$ is simulated first by compute the average profit with current inventory $X_n$, basically the same as generating DTMC before. The profit is calculated by:

$$Profit = \sum_{t=0}^T \beta^t( C_p(Demand-Q)-C_v\cdot order + C_s\cdot oldest -C_h\cdot leftover )$$

Then average N episodes:  $\hat v(x) = \frac{1}{N} \sum_{i=1}^N episode(i)$

At the same time, we also need to calculate the 95% confidence interval:

$$ Confidence\ Interval = [\hat V^L(1)-\epsilon,\ \hat V^L(1)+\epsilon]$$
where $\epsilon=\frac{1.96(\hat \sigma^L)}{\sqrt L}$ with $\hat \sigma^L$ being the sample standard deviation



The results are plotted along with the code, and the data is saved in csv format in another folder. 

The red line is the result predicted by the neural network, and the blue line is the result of previous Monte Carlo  simulation.


The blue shade indicates the confidence interval of Monte Carlo. Since neural network only gives one prediction for a given input, there is no confidence interval for neural network



## 3. Neural Network

The neural network contains one input layer, one hidden layer with 32 neuron, and one output layer with ReLU as its activate function


We will need to use this neural network to predict the expected profit for a given state, and the criterion used here is MSE loss.


The input is 
$$
    \left[
	\begin{matrix}{}
    x_{1000,1} & x_{1000,2} & x_{1000,3} & \cdots & x_{1000,L-1}\\
    x_{2000,1} & x_{2000,2} & x_{2000,3} & \cdots & x_{2000,L-1}\\
    \vdots & \vdots & \vdots & \vdots & \vdots\\    
    x_{1000K,1} & x_{1000K,2} & x_{1000K,3} & \cdots & x_{1000K,L-1}\\
	\end{matrix}
	\right]
$$
where K=20, 200, 2000 respectively

The output is $f_\theta(x)$, where x is each row of this matrix. 

$\tilde v(x)=f_\theta(x)$, and we minimize the mean square loss: $minimize \sum_{k=1}^m (f_\theta(x_k)-\hat v(x_k))^2$



In addition, we needto generate a test dataset, so we run the DTMC for 50000 period, sample ${X_{1000}, X_{2000} \cdots X_{50000}}$ as the test dataset.


The number of epoches for training NN is 100 epoches, and at each 10 epoches, we use these test dataset to evaluate the accuracy of the neural network.



### Question (a)
For question a, our result is:
$$\tilde v(x)=13.8080, 12.7325, 13.5641$$ for neural network trained with K=20, 200, 2000 respectively
$$\hat v(x)=11.2654,\ with\ confidence\ interval\ [10.9702, 11.5936]$$



### Question (b)

The plot is shown along with the code.

The red line is the result predicted by the neural network, and the blue line is the result of previous Monte Carlo  simulation.

The blue shade indicates the confidence interval of Monte Carlo. Since neural network only gives one prediction for a given input, there is no confidence interval for neural network



### Question(c)

```
K=20: mean square error = 0.5433511137962341
```

```
K=200: mean square error = 0.3671260178089142
```

```
K=2000: mean square error = 0.5454710125923157
```