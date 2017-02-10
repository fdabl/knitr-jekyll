---
layout: post
title: "Some Notes on Bayesian Linear Regression"
---



The linear model is the workhorse of applied statistics. It solves the general prediction problem

$$
y = g(X)
$$

by constraining $g(.)$ to be a linear function. The (general) linear model can be written as

$$
\begin{align*}
y_i &\sim \text{Normal}(\mu_i, \sigma_{\epsilon}^2) \\
\mu_i &= \beta_0 + \beta_1 x_i
\end{align*}
$$

The plot below visualizes the linear model and its predictions. The first red pointrange annotation shows the predicted value at a speed of 10; the prediction is perturbed by noise $\sigma_{\epsilon}^2$, the lines indicate the .025 and .975 quantile of the Gaussian distribution (it's not a [prediction interval](https://en.wikipedia.org/wiki/Prediction_interval#Regression_analysis)).

<img src="/figure/source/2017-02-20-Bayesian-regression/unnamed-chunk-2-1.png" title="plot of chunk unnamed-chunk-2" alt="plot of chunk unnamed-chunk-2" style="display: block; margin: auto;" />

Note that the model is linear in the regression weights $\beta$, but need not be linear in
the predictor variables $x$; i.e., the fitted function does not necessarily have to be a
straight line. In the visualisation below, we add the squared speed as an additional predictor.

<img src="/figure/source/2017-02-20-Bayesian-regression/unnamed-chunk-3-1.png" title="plot of chunk unnamed-chunk-3" alt="plot of chunk unnamed-chunk-3" style="display: block; margin: auto;" />

Let's simplify the notation a bit. We can write a series of linear equations as a matrix multiplication, e.g.,

$$
\begin{align*}
1 \cdot \beta_0 + \beta_1 \cdot x_{11} + \beta_2 \cdot x_{12} &= y_1 \\
1 \cdot \beta_0 + \beta_1 \cdot x_{21} + \beta_2 \cdot x_{22} &= y_1 \\
\dots &= \dots \\
1 \cdot \beta_0 + \beta_1 \cdot x_{n1} + \beta_2 \cdot x_{n2} &= y_n \\
\end{align*}
$$

can be written consicely as a matrix multiplication $X\beta = y$ where

$$
\underbrace{\begin{pmatrix}
1 & x_{11} & x_{12} \\
1 & x_{21} & x_{22} \\
\vdots & \vdots & \vdots \\
1 & x_{n1} & x_{n2} \\
\end{pmatrix}}_{X} \cdot
\underbrace{\begin{pmatrix}
\beta_0 \\
\beta_1 \\
\beta_2
\end{pmatrix}}_{\beta} =
\underbrace{\begin{pmatrix}
y_1 \\
y_2 \\
\vdots \\
y_n
\end{pmatrix}}_{y}
$$

The parameters of the model are the vector of the regression weights $\beta$ and the error variance
$\sigma_{\epsilon}^2$. Standard estimation proceeds with using calculus or geometric motivation (see [here](https://en.wikipedia.org/wiki/Linear_regression#Estimation_methods))
to arrive at the point estimates

$$
\hat \beta = (X^T X)^{-1} X^T y
$$

with variance $\text{Var}[\hat \beta] = \sqrt{\sigma_e / N}$. The [Gauss-Markov theorem](https://en.wikipedia.org/wiki/Gauss%E2%80%93Markov_theorem) states that this is
the best linear unbiased estimator if the errors are uncorrelated, have expectation zero, and $\sigma_{\epsilon}^2$ is constant across levels of the predictor (more on that below). However, we might want to trade some unbiasedness for increased prediction accuracy, say in high-dimensional settings, and thus might prefer the lasso or ridge regression (see [here](https://en.wikipedia.org/wiki/Regularization_(mathematics)) and [here](https://stats.stackexchange.com/questions/866/when-should-i-use-lasso-vs-ridge)).

In this post, we focus on a Bayesian solution. Bayesian statistics entails quantifying one's uncertainty about the world using probability. All further inference reduces simply to applying the rules of standard probability theory. Based on the *product rule* and the *sum rule* (e.g., Lindley, [2000](http://www.phil.vt.edu/dmayo/personal_website/Lindley_Philosophy_of_Statistics.pdf))

$$
\begin{split}
p(\mathbf{\theta}, \textbf{y}) &= p(\textbf{y}|\theta) p(\theta) \\
p(\textbf{y})             &= \int p(\mathbf{\theta}, \textbf{y}) \mathrm{d}\mathbf{\theta}
\end{split}
$$

which follow from the [axioms of probability theory](https://en.wikipedia.org/wiki/Probability_axioms), we can derive Bayes' rule

$$
\begin{split}
p(\textbf{y}, \mathbf{\theta}) &= p(\textbf{y}, \mathbf{\theta}) \\
p(\mathbf{\theta}|\textbf{y})p(\textbf{y}) &= p(\textbf{y}|\mathbf{\theta})p(\textbf{y}) \\[1.5ex]
\underbrace{p(\mathbf{\theta}|\textbf{y})}_{\text{Posterior}} &= \frac{p(\textbf{y}|\mathbf{\theta})p(\mathbf{\theta})}{p(\textbf{y})} = 
\frac{\overbrace{p(\textbf{y}|\mathbf{\theta})}^{\text{Likelihood}}\overbrace{p(\mathbf{\theta})}^{\text{Prior}}}{\underbrace{\int p(\textbf{y}|\mathbf{\theta})p(\mathbf{\theta}) \mathrm{d}\mathbf{\theta}}_{\text{Marginal Likelihood}}}
\end{split}
$$

where $\mathbf{\theta}$ denotes our parameter vector and $\textbf{y}$ denotes the data we have collected. In theory, this is all there is to Bayesian statistics. In practice, of course, things are more intricate. First, we must specify the likelihood which links the data to the parameter vector, $p(\textbf{y}\|\mathbf{\theta})$, commonly called the statistical model --- this will be the focus of this blog post. Second, we must specify a prior probability over our parameter vector, $p(\mathbf{\theta})$, which is a novum compared to classical statistics. Third, we must apply Bayes' rule, which involves computing a possibly high-dimensional integral. For the latter task, we will utilize the probabilistic programming language [Stan](http://mc-stan.org/).

For the linear model, Bayes' rule amounts to

$$
\begin{align*}
p(\beta, \sigma_{\epsilon}|X, \textbf{y}) &= \frac{p(\textbf{y}|X, \beta, \sigma_{\epsilon}) p(\beta) p(\sigma_{\epsilon})}
                                             {\int \int p(\textbf{y}|X, \beta, \sigma_{\epsilon}) p(\beta) p(\sigma_{\epsilon}) \mathrm{d}\beta \mathrm{d}\sigma_{\epsilon}}
\end{align*}
$$

This looks daunting, but it's just looks. We assume that $\beta$ and $\sigma_{\epsilon}$ are independent so that we can write $p(\beta, \sigma_{\epsilon}) = p(\beta) p(\sigma_{\epsilon})$. There is an analytical solution to this problem if we assume Gaussian priors for $\beta$ (for the mathematics, see [here](https://en.wikipedia.org/wiki/Bayesian_linear_regression#Conjugate_prior_distribution)).

Note that we assume *weak exogeneity*, i.e., $\mathbb{E}[X] = X$, or that the predictor variables a free of measurement error. This is quite an assumption, as Westfall & Yarkoni ([2016](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0152719)) have demonstrated.

## Small example
We write a convenience function that just takes the dependent variable and the predictor variables and a Stan model string and estimates the model.


{% highlight r %}
library('rstan')
options(mc.cores = parallel::detectCores())

run_model <- function(y, X, ms, ...) {
  stan_dat <- list('y' = y, 'X' = X, 'N' = length(y), 'p' = nrow(X))
  fit <- stan(
    data = stan_dat,
    model_code = ms,
    ...
  )
  fit
}
{% endhighlight %}

The Stan specification of the linear model is


{% highlight r %}
ms_lm <- '
data {
  int<lower = 1> p;
  int<lower = 1> N;
  real y[N];
  matrix[N, p] X;
}

parameters {
  vector[p] beta;
  real<lower = 0, upper = 20> sigma_e;
}

model {
  beta ~ normal(0, 1);
  sigma_e ~ uniform(0, 20);
  y ~ normal(X * beta, sigma_e); # vectorized!
}

generated quantities {
  real y_pred;
  y_pred <- normal_rng(X * beta, sigma_e);
}
'
{% endhighlight %}

The corresponding graphical model can be seen in Figure 2.


{% highlight r %}
with(
  cars,
  y <- dist,
  X <- cbind(1, scale(speed, scale = FALSE))
)
m <- run_model(y, X, ms_lm)
{% endhighlight %}

Regularization is built into the Bayesian framework. For example, using the mode of the posterior distribution as our point estimate  --- commonly called the MAP estimate --- linear regression with Gaussian priors equals ridge regression (see [here](https://stats.stackexchange.com/questions/163388/l2-regularization-is-equivalent-to-gaussian-prior)), while using Laplace priors recovers the lasso (Park & Casella, [2008](http://www.stat.ufl.edu/archived/casella/Papers/Lasso.pdf)).

## Generalized Linear model
Not all dependent variables are continuous. We might be interested in number of clicks on an online ad (Poisson),
probability of dying in clinical trials (Bernoulli), or school grades (Rank). There is an ingenious trick to model these data without abandoning the linear modeling framework. Specifically, one maps the linear predictor --- in our case above $X\beta$ --- onto the domain of the dependent variable.


As an example on which we will elaborate more below, take logistic regression (using matrix multiplication for the linear equation part). It's specificaiton is

$$
\begin{align*}
y_i &\sim \text{Bernoulli}(p_i) \\
f(p_i) &= X_{i.} \beta
\end{align*}
$$

where $f(.)$ is called the *link function*, and $X_{i.}$ denotes the $i^{\text{th}}$ row of $X$. Note that we model the probability of a success, $p_i$, directly. This requires a link function which maps the continous linear predictor onto the domain [0, 1]. This is vital for prediction, because probabilities below 0 or above 1 do not exist; however, the linear predictor does not know that, as is apparent in the plot below.


{% highlight r %}
N <- 10000
Xb_TeX <- '$X\\beta$'
dat <- data.frame(y = seq(0, 1.3, length.out = N), x = seq(0, 10, length.out = N))

find_intersect <- function(dat) {
  row <- head(which(abs(dat$y - 1) < .0001), 1)
  dat$x[row]
}

ggplot(dat, aes(x = x, y = y)) +
  geom_line(colour = 'skyblue', linetype = 'longdash') +
  geom_hline(yintercept = 1, color = 'red', linetype = 'dashed') +
  geom_segment(x = find_intersect(dat), xend = Inf, y = 1, yend = 1, color = 'skyblue') +
  scale_x_continuous(breaks = scales::pretty_breaks(10)) +
  ylab('Probability') +
  xlab(TeX(Xb_TeX)) +
  ggtitle('Motivation for a link function') +
  theme(plot.title = element_text(hjust = .5)) +
  annotate('rect', xmin = -Inf, xmax = Inf, ymin = 1, ymax = Inf, alpha = .2, fill = 'red')
{% endhighlight %}

<img src="/figure/source/2017-02-20-Bayesian-regression/unnamed-chunk-7-1.png" title="plot of chunk unnamed-chunk-7" alt="plot of chunk unnamed-chunk-7" style="display: block; margin: auto;" />

### Logistic regression
For example, assume a Bernoulli random variable, i.e., a random variable who is either 0 or 1. Our linear predictor, however, is on the continuous domain; we require a mapping between the two, which will be $f^{-1}$, the *inverse link function*. We can use a Sigmoid function to map from the continuous domain of the linear predictor, $Xb$, onto the domain [0, 1]; see the figure below.

<img src="/figure/source/2017-02-20-Bayesian-regression/unnamed-chunk-8-1.png" title="plot of chunk unnamed-chunk-8" alt="plot of chunk unnamed-chunk-8" style="display: block; margin: auto;" />

Mathematically, the Sigmoid function is

$$
f(X\beta) = \frac{1}{1 + e^{-X\beta}}
$$

We arrive at the inverse link function $f^{-1}$ by solving

$$
\begin{align*}
y &= \frac{1}{1 + e^{-X\beta}} \\
(1 + e^{-X\beta})y &= 1 \\
e^{-X\beta} &= \frac{1 - p}{p} \\
X\beta &= \log \left( \frac{p}{1 - p} \right)
\end{align*}
$$

This inverse link function is the *log odds ratio*; that's nice! You might be familiar with the odds ratio from betting. Assume the probability of Germany winning the next soccer world cup is $p = .2$. This means it has odds of winning of $\frac{.2}{1 - .2} = .25$, which is not too bad. For every Euro I bet on Germany, I should get (more than) four in return for me to make that bet.

For the binomial case, we similarly apply Bayes' rule, noting that the likelihood term differs and that
we do not have a error variance

$$
p(\beta|X, \textbf{y}) = \frac{p(\textbf{y}|f^{-1}(X\beta)) p(\beta)}
                          {\int p(\textbf{y}|f^{-1}(X\beta)) p(\beta) \mathrm{d}\beta}
$$

This time, we have no closed form solution --- but we have Stan.

Let's take a small example data set and estimate the model in Stan, pointing out a common error in data analysis. Frequently, people apply a linear model (e.g., ANOVA) to averaged Bernoulli trials. For example, assume you want to know whether participants make more errors in condition A than in condition B. The most common approach is to calculate, for each participant, the percentage of correct trials and model this percentage.

However, because the underlying variable is Bernoulli, using an ANOVA results --- per definition --- in a violation of homoscedasticity, i.e., the variance is not constant across different levels of the predictor (which, in this case, is just the mean). To make this apparent, let's look at the Bernoulli distribution

$$
f(x|p) = p^x (1 - p)^x
$$

the mean or expectation of this distribution is $\mathbb{E}[X] = p$, and the variance is $\mathbb{Var}[X] = p (1 - p)$. In contrast to the Gaussian distribution, the variance is not independent of the mean, as can be seen in the plot below.


<img src="/figure/source/2017-02-20-Bayesian-regression/unnamed-chunk-9-1.png" title="plot of chunk unnamed-chunk-9" alt="plot of chunk unnamed-chunk-9" style="display: block; margin: auto;" />

Homoscedasticity, then, is only the case should conditions A and B have the same $p$, or be equidistant from $p = .5$. For more on this topic, see J?ger ([2008](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2613284/)). Below we run a linear model on percent correct, and then a logistic regression on a trial basis.


{% highlight r %}
ms_bin <- '
data {
  int<lower = 1> p;
  int<lower = 1> N;
  int<lower = 0, upper = 1> y[N];
  matrix[N, p] X;
}

parameters {
  vector[p] beta;
}

model {
  beta ~ normal(0, 1);
  theta = inv_logit(X * beta); # g: [-Inf, Inf] -> [0, 1]
  y ~ bernoulli(theta);
}
'
{% endhighlight %}

After we have updated our beliefs over $\beta$, and in order to talk about changes in our predictors influencing the *probability of success*, we have to transform the predictors back using the *log odds ratio*.

## Ordered regression
This is the topic I set out to write about in the first place. It took me a while to get there, but I think we
are now well prepared to tackle it.

Many people treat questionnare data as being on an interval scale; that is, assuming that differences are
equidistant; i.e., ticking 2 instead of 1 is the same as ticking 6 instead of 5.

However, while this is already a bold assumption, there are cases where this is flat out wrong, say in school grades.
There, the difference between a 2 and a 1 is much bigger than between a 4 and a 5 (assuming German grades).

In addition to that, the response variable is bounded by the scale, that is --- per definition --- not a Gaussian.

Relying on the trick we have learned above, we need to find a (inverse) link function that maps our linear predictor
onto the domain of the response. In addition, we want to respect the assumption that the response is *rank ordered*; i.e.,
a 1 is better than a 2, a 2 better than a 3 etc.

In the logistic regression case above, we have already made contact with the odds ratios and Sigmoid functions for $k = 2$ categories. We can generalize these mathematical tools to $k > 2$ cases, i.e., for a multinomial response, which is what we require for ordinal regression. Ordinal regression with $k = 2$ is logistic regression.


{% highlight r %}
ms_ord <- '
data {
  int<lower = 1> p;
  int<lower = 1> N;
  int<lower = 1> K;
  int<lower = 0, upper = 1> y[N];
  matrix[N, p] X;
}

parameters {
  ordered[K-1] c;
  vector[p] beta;
}

model {
  beta ~ normal(0, 1);
  y ~ ordered_logistic(X * beta, c)
}
'
{% endhighlight %}

## Further reading
For an easy introduction and overview of Bayesian statistics, see Etz et al. ([2016](https://osf.io/8wkpd/)). For a discussion of regularization, see [this]. For very thoughtful comments on maximum entropy and the generalized linear model, see chapter 9 of McElreath (2016). For more on Stan, see their [excellent documentation](http://mc-stan.org/documentation/).

A future blog post might look into hierarchical extensions of the models discussed here.
