# Recommender Systems using Matrix Factorization

## Problem Statement & Setup :
Matrix factorization is a way to generate low rank matrix when multiplying two differentmatrices, and thanks to that we can assimilate missed reviews, derived by examining theassociations between the users and the items. Hence, we can predict if a user is likely to givea movie high rating or low rating, and given that information the system recommend themovie to that user.

![enter image description here](https://i.imgur.com/6ja4LSX.png)
Our goal is to find the matrices <img src="https://latex.codecogs.com/gif.latex?U" />  and <img src="https://latex.codecogs.com/gif.latex?I" />  which estimates as close as possible the matrix <img src="https://latex.codecogs.com/gif.latex?R" />  , in other words, we want to minimize a function <img src="https://latex.codecogs.com/gif.latex?C" /> 

![equation](https://latex.codecogs.com/gif.latex?\min&space;_{I,&space;U}\left\|R-I&space;U^{\top}\right\|_{\mathcal{F}}^{2}&plus;\lambda\|I\|_{\mathcal{F}}^{2}&plus;\mu\|U\|_{\mathcal{F}}^{2} )

## Proposed Solutions :
Our Pipeline can be summarized as follows : 
![enter image description here](https://i.imgur.com/lOwwdpu.png)

### Take away : 
In this project we addressed the problem of Matrix factorization, where we implemented two approaches to solve it which are : stochastic gradient descent and Alternating least squares.Through this study, we can conclude that stochastic gradient descent with biases is a good solution for this problem, since it gives better results and takes less running time.
