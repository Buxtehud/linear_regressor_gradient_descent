# Linear Regressor Gradient Descent

## Theory

This is a Linear Regression model implementing Gradient Descent. Just made for fun and experimenting with machile learning algorithms.

First, I'll select the sum of squared residuals as the loss function:

$$\sum_{i=0}^{n} (\hat{y}_i - y_i)^2$$

Where $\hat{y}_i$ represent the data points and $y_i$ the predicted values.

Our predicted value is a straight line, so we get:

$$y=mx+b$$

Where $m$ is our slope and $b$ our intercept.

With these we get:

$$\sum_{i=0}^{n} [\hat{y}_i - (b + mx_i)]^2$$

To find the optimal values, we want to minimize our  loss function. To do so, using gradient descent, we calculate the gradient.

Given a two variable function $f(x,y)$, the gradient is:

$$\vec{\nabla} f = \left(\frac{\partial{f}}{\partial{x}}, \frac{\partial{f}}{\partial{y}} \right)$$

To aproximate numerically the derivatives, we will use finite differences:

$$\frac{\partial{f}}{\partial{x}} = \frac{f(x+\Delta h, y) - f(x-\Delta h, y)}{2\Delta h} + O(h^2)$$
$$\frac{\partial{f}}{\partial{y}} = \frac{f(x, y+\Delta h) - f(x, y-\Delta h)}{2\Delta h} + O(h^2)$$

Where $\Delta h$ is the step and $O(h^2)$ the error.

I use central differences because that give us better aproximations than forward or backward differences and our function is $C^{\infty}$ and do not oscilate, so we won't have mayor difficulties.

Now, what really matters:

The gradient descent algorithm looks to minimize the loss function. To do so, it iterates through values to find the minimum of such function.

The minimun or maximun of a multivariable function is where $\vec{\nabla} f = (0,0)$.

We know for sure that our loss function

$$L(b,m) = \sum_{i=0}^{n} [\hat{y}_i - (b + mx_i)]^2$$

has a minimum because is convex.

So, we need to find where his gradient is zero.

To do so, first we calculate the gradient given an arbitrary intercept $b_0$ and an arbitrary slope $m_0$

$$ \vec{\nabla} L(b_0, m_0) = (g_b^0, g_m^0)$$

Then we multiply our gradient by an arbitrary small value, called learn rate ($l_r$), to get the step size(s) of each variable (slope and intercept)

$$s_b =  g_{b_0}\times l_r$$
$$s_m =  g_{m_0}\times l_r$$

The new values of our slope and intercept will be:

$$m_1 = m_0 - s_m$$
$$b_1 = b_0 - s_b$$

Then we calculate the new gradient with the new slope and intercept and so on, until the step size gets really small. How much is an hyperparameter that we can adjust.

So, our values are:

$$ m_i = m_{i-1} - l_r \nabla L_m(b_{i-1}, m_{i-1}) $$
$$ b_i = b_{i-1} - l_r \nabla L_b(b_{i-1}, m_{i-1}) $$

Also notice that the derivative operator is linear. That's imortant, because our loss function is a sum of multiple terms, and we can calculate its derivative easily by calculating the derivative at each point and then adding them all togheter:

$$ \vec{\nabla} L = \left(\frac{\partial}{\partial m}\sum_{i=0}^{n} [\hat{y}_i - (b + mx_i)]^2,\frac{\partial}{\partial b}\sum_{i=0}^{n} [\hat{y}_i - (b + mx_i)]^2\right) $$
$$ \vec{\nabla} L = \left(\sum_{i=0}^{n} \frac{\partial}{\partial m}[\hat{y}_i - (b + mx_i)]^2,\sum_{i=0}^{n} \frac{\partial}{\partial b}[\hat{y}_i - (b + mx_i)]^2\right) $$

## Test

![gradient_descent_plot1](https://user-images.githubusercontent.com/73806621/196062127-f1ee9c62-f8cf-4c55-a9ff-fc77d21a490d.png)
![gradient_descent_plot2](https://user-images.githubusercontent.com/73806621/196062130-6686ba41-4418-4cbf-8469-5ef6419b1c8c.png)

Comparing my results with those given by Scikit-Learn, it's pretty descent. It needs to be optimized a lot, and as it has been done with python, is not so good perfomance wise.
