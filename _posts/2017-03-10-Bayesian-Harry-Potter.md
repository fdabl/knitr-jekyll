---
layout: post
title: "Harry Potter and Bayesian order-constrained inference"
---



In recent weeks, I have started to say "You're such a Hufflepuff!" to a girl I live with in my student apartment; but what do I mean by that? What is "being a Hufflepuff" a proxy for?

In terms of sales, Harry Potter is a franchise as big as Star Wars or Lord of the Rings. It has captured the imagination of a wide audiences across all age groups. Recently, J.K. Rowling has put together a few online questions that, upon answering them, sort you into one of the four Harry Potter Houses: Gryffindor, Ravenclaw, Slytherin, Hufflepuff.


[PICTURE OF POTTERMORE]

We have just submitted a paper on the relation between the Harry Potter Houses and measures of personality and human values. Briefly, we replicate and extend previous findings showing that people who have been assigned to a certain House, e.g., Slytherin, score higher on traits that are associated with that House, e.g., Psychopathy, Machiavellianism, Narcissism (see the figure below). All materials can be found at [osf.io/url](osf.io/url).



<img src="/figure/source/2017-03-10-Bayesian-Harry-Potter/unnamed-chunk-3-1.png" title="Boxplots for the Big Five and Dark Triad scores across Houses." alt="Boxplots for the Big Five and Dark Triad scores across Houses." style="display: block; margin: auto;" />

## The statistical problem
Using Psychopathy as an example, all of our hypotheses were of the following form

$$
H_{\text{Psychopathy}} : \mu_{\text{S}} > (\mu_{\text{G}} , \, \mu_{\text{R}} , \, \mu_{\text{H}})
$$

where $\mu_x$ indicates the mean of the measure for the respective House. Framed differently, our hypothesis was that Slytherin will *score highest* on Psychopathy compared to the other Houses. Testing such hypotheses is difficult in classical statistics.

In this blog post, I want to show how we have tackled this problem. To this end, I provide a brief introduction to Bayesian inference, show how to compute Bayes factors for one-way ANOVAs using Cauchy priors, and discuss the advantages of testing order constrained hypotheses.


## Bayesian inference 101
Bayesian statistics entails quantifying one's uncertainty about the world using probability. Statistical inference reduces simply to applying the rules of standard probability theory. For an introduction to Bayesian inference from this angle, see Etz & Vandekerckhove ([2017](https://osf.io/preprints/psyarxiv/q46q3)).

The method dictated by probability theory for Bayesian hypothesis testing is the Bayes factor. Let the null hypothesis be instantiated by a restricted model, $M_0$, while the alternative hypothesis is specified by $M_1$. Parameters within the models --- here denoted by $\theta$ --- are estimated using Bayes' rule

$$
p(\mathbf{\theta}|\textbf{y}, M_0) = \frac{p(\textbf{y}|\mathbf{\theta}, M_0)p(\mathbf{\theta}|M_0)}{\int p(\textbf{y}|\mathbf{\theta}, M_0)p(\mathbf{\theta}|M_0) \mathrm{d}\theta}
$$

An equivalent expression exists for $M_1$. Note that, using the *sum rule* of probability, the denominator can be written as $p(\textbf{y}\|M_0)$, which is sometimes called the *marginal likelihood*.

The model selection agenda is to test whether $M_0$ or $M_1$ is more probable, given the collected data. Assuming simplifyingly that $M_0$ and $M_1$ exhaust the model space, i.e., $p(M_0) + p(M_1) = 1$, we apply Bayes' rule

$$
\begin{split}
p(M_0|\textbf{y}) &= \frac{p(\textbf{y}|M_0)p(M_0)}{p(\textbf{y})} \\
p(M_1|\textbf{y}) &= \frac{p(\textbf{y}|M_1)p(M_1)}{p(\textbf{y})}
\end{split}
$$

Taking ratios we arrive at

$$
\begin{split}
\frac{p(M_0|\textbf{y})}{p(M_1|\textbf{y})} &= \frac{p(\textbf{y}|M_0)}{p(\textbf{y}|M_1)}\frac{p(M_0)}{p(M_1)} \\[1ex]
\text{Posterior odds} &= \text{Bayes factor} \cdot \text{Prior odds} \\[1ex]
\text{Bayes factor} &= \frac{\text{Posterior odds}}{\text{Prior odds}}
\end{split}
$$

Therefore, the Bayes factor numerically denotes our shift in the models' beliefs given the data.


### Bayesian ANOVA
Analysis of variance (ANOVA), is the workhorse of experimental psychology


### Order-restricted Bayes factors
How does this fit with our model selection problem? Note that the Bayes factor is transitive, such that

$$
\frac{p(\textbf{y}|M_r)}{p(\textbf{y}|M_0)} = \frac{p(\textbf{y}|M_r)}{p(\textbf{y}|M_f)} \cdot \frac{p(\textbf{y}|M_f)}{p(\textbf{y}|M_0)}
$$




{% highlight r %}
library('BayesFactor')

compute_BFs <- function(formula, main_house, dat, iterations = 10000, r = 1/2) {
  
  BFf0 <- anovaBF(formula, data = dat, progress = FALSE,
                  rscaleEffects = c('Sorting_house' = r))
  
  houses <- c('Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff')
  other_houses <- houses[main_house != houses]

  # Get samples from the posterior distribution
  post <- posterior(BFf0, iterations = iterations)
  header <- colnames(post)
  
  # Mean of the main house (sum to zero coded)
  main <- post[, grepl(main_house, header)]
  
  # Mean of the other houses (sum to zero coded)
  other_house_pattern <- paste(other_houses, collapse = '|')
  other <- post[, grepl(other_house_pattern, header)]
  
  prior_odds <- 1 / factorial(4)
  posterior_odds <- mean(apply(main > other, 1, all))
  
  BFf0 <- extractBF(BFf0)$bf
  BFrf <- posterior_odds / prior_odds
  BFr0 <- BFrf * BFf0
  
  log(c(BFf0, BFr0, BFrf))
}

compute_BFs(SD3_Psychopathy ~ Sorting_house, 'Slytherin', dat, r = 1/2)
{% endhighlight %}

[1] 21.984966 25.163020  3.178054
