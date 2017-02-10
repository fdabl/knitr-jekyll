---
layout: post
title: "Analysis for our Registered Replication of Cohen et al. (2015)"
author: "Fabian Dablander, Katharina Brecht, Lea Jakob, Nicola Clayton"
---



# Introduction
In this document we go through and discuss our planned analysis for our Registered Replication of Cohen et al. (2015). We simulate data according to our hypotheses, and setup R code to automatically analyze the resulting data. Therefore, once we have finished our own data collection, everything is already automated.


{% highlight r %}
library('afex')
library('lme4')
library('papaja')
library('tidyverse')
library('BayesFactor')
theme_set(theme_apa())
{% endhighlight %}

# Simulation setup
Let's simulate data according to our hypotheses. Let 'P' and 'Q' denote our within-subject conditions, i.e., 'truth value' and 'representation (mental vs. non-mental note)', and let 'B' denote our between-subject condition, i.e., 'instruction'.


{% highlight r %}
simulate_dat <- function(N, trials) {
  P <- 2
  Q <- 2 ## P, Q: within factors
  B <- 2 # B: between factor
  
  d <- data.frame(
    id = rep(seq(N), each = P*Q*trials),
    trial = rep(seq(trials), each = P*Q),
    B = rep(seq(2) - 1, each = N*trials*P*Q / 2)
  )
  
  i <- 0
  n_total <- N*trials
  within <- expand.grid(P = seq(P), Q = seq(Q)) - 1
  within_mat <- matrix(NA, nrow = n_total, ncol = 2)

  # here we randomly assign the PxQ within conditions
  while (i < n_total) {
    w <- sample(within)
    for (k in seq(4)) {
      for (j in seq(2)) {
        within_mat[k + i, j] <- w[k, j]
      }
    }
    
    i <- i + 4
  }
  
  colnames(within_mat) <- c('P', 'Q')
  cbind(d, within_mat) %>% 
    select(id, trial, P, Q, B)
}

d <- simulate_dat(100, 20) # for computational efficiency, only 100 participants and 20 trials
head(d)
{% endhighlight %}



{% highlight text %}
##   id trial P Q B
## 1  1     1 0 0 0
## 2  1     1 1 0 0
## 3  1     1 0 1 0
## 4  1     1 1 1 0
## 5  1     2 0 0 0
## 6  1     2 1 0 0
{% endhighlight %}

Therefore, in the following we assume that there is no main effect of representation on response time and errors, which corresponds to $H_{1a}$ and $H_{1b}$. Additionally, we assume that there is no effect of instruction on representation, i.e., no interaction, which corresponds to our $H_2$.

### Reaction time data
Most frequently, reaction times do not follow a normal distribution. Instead, recent research suggests assuming a Gamma distribution works best [Lo and Andrews (2015)](http://journal.frontiersin.org/article/10.3389/fpsyg.2015.01171/full).
  
The pdf of the Gamma distribution is
$$
f(x; \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1} e^{-\beta x}
$$

with a mean of $\mathbb{E}[X] = \frac{\alpha}{\beta}$ where $\alpha$ is the shape and $\beta$ is the rate parameter. In the simulations, we draw from two Gamma distributions, one with $\alpha = 10, \beta = 1/30$, the other with $\alpha = 9, \beta = 1/30$ such that the mean effect of 'representation' is $300 - 270 = 30$ milliseconds.

### Error data
On each trial, whether the partipant makes an error or not, is modeled as a Bernoulli distribution

$$
f(x; \theta) = \theta^x (1 - \theta)^{1 - x}
$$

where $\theta$ determines the likelihood of an error. For a trial with a mental representation, we assume $\theta = .5$, i.e., random errors, while a non-mental representation should have more errors, say $\theta = .75$.


{% highlight r %}
set.seed(1774)

nn <- nrow(d) / 2
dat <- d %>% 
  rename(
    repr = Q,
    truth = P,
    instr = B
  ) %>% 
  mutate(
    RT = ifelse(repr == 0, rgamma(nn, shape = 10, rate = 1/30), rgamma(nn, shape = 9, rate = 1/30)),
    err = ifelse(repr == 0, rbinom(1, n = nn, prob = .75), rbinom(1, n = nn, prob = .5))
  )
{% endhighlight %}

Before we visualize the simulation results, we have to apply our outlier selection criteria. 


{% highlight r %}
datf <- dat %>% 
  filter(
    RT < mean(RT) + 3*sd(RT),
    RT > mean(RT) - 3*sd(RT),
    err < mean(err) + 3*sd(err),
    err > mean(err) - 3*sd(err)
  ) %>% 
  mutate(
    truth = factor(ifelse(truth == 0, 'false', 'true')),
    repr = factor(ifelse(repr == 0, 'non-mental', 'mental')),
    instr = factor(ifelse(instr == 0, 'Old instruction', 'New instruction'))
    )
{% endhighlight %}

This lead to the removal of

{% highlight r %}
((nrow(dat) - nrow(datf)) / nrow(dat)) * 100
{% endhighlight %}



{% highlight text %}
## [1] 0.65
{% endhighlight %}
percent of the data. Additionally, for the reaction time data, we remove trials that the participants got wrong.

{% highlight r %}
dat_rt <- filter(datf, err != 1)
{% endhighlight %}

### Data visualization

{% highlight r %}
dat_errm <- datf %>% 
  group_by(repr, truth) %>% 
  summarize(err = mean(err))

dat_rtm <- dat_rt %>% 
  group_by(repr, truth) %>% 
  summarize(meanRT = mean(RT))

dat_errm
{% endhighlight %}



{% highlight text %}
## Source: local data frame [4 x 3]
## Groups: repr [?]
## 
##         repr  truth       err
##       <fctr> <fctr>     <dbl>
## 1     mental  false 0.4783484
## 2     mental   true 0.4969940
## 3 non-mental  false 0.7467205
## 4 non-mental   true 0.7510081
{% endhighlight %}



{% highlight r %}
dat_rtm
{% endhighlight %}



{% highlight text %}
## Source: local data frame [4 x 3]
## Groups: repr [?]
## 
##         repr  truth   meanRT
##       <fctr> <fctr>    <dbl>
## 1     mental  false 262.9063
## 2     mental   true 267.3407
## 3 non-mental  false 293.5020
## 4 non-mental   true 297.0942
{% endhighlight %}

Below we plot the data as predicted; averaged across participants.

{% highlight r %}
datsum_err <- datf %>% 
  group_by(instr, repr, truth) %>% 
  summarize(err = mean(err))

datsum_rt <- dat_rt %>% 
  group_by(instr, repr, truth) %>% 
  summarize(meanRT = mean(RT))

ggplot(datsum_rt, aes(x = repr, y = meanRT, color = truth)) +
  geom_point() +
  geom_line(aes(group = truth)) +
  facet_wrap(~ instr) +
  ylim(c(200, 300)) +
  xlab('Representation') +
  ylab('Mean RT') +
  ggtitle('Predictions Reaction Time (averaged)') +
  theme(plot.title = element_text(hjust = .5))
{% endhighlight %}

![plot of chunk unnamed-chunk-8](/figure/source/2017-01-21-registered-report-analysis/unnamed-chunk-8-1.png)

{% highlight r %}
ggplot(datsum_err, aes(x = repr, y = err, color = truth)) +
  geom_point() +
  geom_line(aes(group = truth)) +
  facet_wrap(~ instr) +
  ylim(c(0, 1)) +
  xlab('Representation') +
  ylab('Mean % error') +
  ggtitle('Predictions Error rate (averaged)') +
  theme(plot.title = element_text(hjust = .5))
{% endhighlight %}

![plot of chunk unnamed-chunk-8](/figure/source/2017-01-21-registered-report-analysis/unnamed-chunk-8-2.png)

Now that we have simulated our data, let's proceed with the planned analysis.

# Data analysis
Note that the data is unbalanced due to our outlier removal, and thus the assumption of ANOVA are violated. While I proceed with focusing on ANOVA based analyses, I recommend the generalized linear mixed model (GLMM) approach. Because GLMMs do not converge easily using classical estimation, I would utilize Stan to estimate it in a Bayesian fashion, and focus on model checking and computing effect sizes based on the fitted model.

Additionally, for the error dependent variable, averaging across participants and computing an ANOVA on percent change is suboptimal. Because errors represent choice data, thus following a Bernoulli distribution, the variance is $\mathrm{Var}[X] = \theta (1 - \theta)$, i.e., not constant for values of $\theta$. This violates homoscedasticity, a crucial assumption of ANOVA [see Jäger (2008)](http://www.sciencedirect.com/science/article/pii/S0749596X07001337). The figure below makes this apparent. The variance of the groups is only equal should $\theta = .5$, or when they are equidistant from $\theta = .5$, which we cannot know a priori.


{% highlight r %}
theta <- seq(0, 1, length.out = 100)
var_bern <- function(theta) theta * (1 - theta)
dat_bern <- data.frame(variance = var_bern(theta), theta = theta)

ggplot(dat_bern, aes(x = theta, y = variance)) +
  geom_line()
{% endhighlight %}

![plot of chunk unnamed-chunk-9](/figure/source/2017-01-21-registered-report-analysis/unnamed-chunk-9-1.png)

As apparent on the plot below, the variance of two groups will be equal only when their $\theta$ is the same, or equidistant from $\theta = .5$. Because this cannot be determined a priori, most often homoscedasticity is violated.

Therefore, the optimal model would be a generalized linear mixed model using a Bernoulli link function.


## Bayesian Generalized Linear Mixed Models
As pointed out above, however, reaction times are rarely normally distributed. Therefore, below, we assume a Gamma distribution for reaction times. For the error data, we assume a Bernoulli distribution.

Further, we use a maximal random effects structure as specified by the design (cf., Barr et al., 2013)

{% highlight r %}
library('rstanarm')
{% endhighlight %}

### Reaction time

{% highlight r %}
m_full <- stan_glmer(RT ~ truth*repr*instr + (truth*repr|id),
                     family = Gamma(link = log), dat = dat_rt,
                     cores = 2, iter = 300, chains = 2)

m_simple <- stan_glmer(RT ~ repr + (truth*repr|id),
                       family = Gamma(link = log), dat = dat_rt,
                       cores = 2, iter = 300, chains = 2)
{% endhighlight %}

### Errors

{% highlight r %}
m_full_err <- stan_glmer(err ~ truth*repr*instr + (truth*repr|id),
                         family = binomial(link = 'logit'), dat = datf,
                         cores = 2, iter = 300, chains = 2)

m_simple_err <- stan_glmer(err ~ repr + (truth*repr|id),
                           family = binomial(link = 'logit'), dat = datf,
                           cores = 2, iter = 300, chains = 2)
{% endhighlight %}

# Appendix
## Frequentist Generalized Linear Mixed models
As pointed out above, however, reaction times are rarely normally distributed. Therefore, below, we assume a Gamma distribution for reaction times. For the error data, we assume a Bernoulli distribution.

Further, we use a maximal random effects structure as specified by the design (cf., Barr et al., 2013)

### Reaction time

{% highlight r %}
m_rt <- glmer(RT ~ truth*repr*instr + (truth*repr|id), family = Gamma, data = dat_rt)

# correlation between fixed effects is not of interest to us
print(summary(m_rt), correlation = FALSE)
{% endhighlight %}

### Errors

{% highlight r %}
m_err <- glmer(err ~ truth*repr*instr + (truth*repr|id), family = binomial, data = datf)

print(summary(m_err), correlation = FALSE)
{% endhighlight %}

## Frequentist Split-plot ANOVA
### Reaction time

{% highlight r %}
m_rt <- aov_car(RT ~ truth*repr*instr + Error(id/(truth*repr)), data = dat_rt)
summary(m_rt)
{% endhighlight %}



{% highlight text %}
## 
## Univariate Type III Repeated-Measures ANOVA Assuming Sphericity
## 
##                        SS num Df Error SS den Df          F    Pr(>F)
## (Intercept)      31480469      1   175391     98 17589.7644 < 2.2e-16
## instr                   0      1   175391     98     0.0000    1.0000
## truth                1115      1   118323     98     0.9234    0.3390
## instr:truth             0      1   118323     98     0.0000    1.0000
## repr                95706      1   109808     98    85.4147 5.347e-15
## instr:repr              0      1   109808     98     0.0000    1.0000
## truth:repr             69      1   137168     98     0.0490    0.8252
## instr:truth:repr        0      1   137168     98     0.0000    1.0000
##                     
## (Intercept)      ***
## instr               
## truth               
## instr:truth         
## repr             ***
## instr:repr          
## truth:repr          
## instr:truth:repr    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
{% endhighlight %}

### Errors

{% highlight r %}
m_err <- aov_car(err ~ truth*repr*instr + Error(id/(truth*repr)), data = datf)
summary(m_err)
{% endhighlight %}



{% highlight text %}
## 
## Univariate Type III Repeated-Measures ANOVA Assuming Sphericity
## 
##                       SS num Df Error SS den Df          F Pr(>F)    
## (Intercept)      152.772      1   1.1282     98 13270.8104 <2e-16 ***
## instr              0.000      1   1.1282     98     0.0000 1.0000    
## truth              0.014      1   1.2253     98     1.1299 0.2904    
## instr:truth        0.000      1   1.2253     98     0.0000 1.0000    
## repr               6.829      1   1.0467     98   639.4073 <2e-16 ***
## instr:repr         0.000      1   1.0467     98     0.0000 1.0000    
## truth:repr         0.005      1   1.1549     98     0.4294 0.5138    
## instr:truth:repr   0.000      1   1.1549     98     0.0000 1.0000    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
{% endhighlight %}

## Bayesian Split-plot ANOVA
Below we use the 'BayesFactor' package to compute the Bayes factor for our split-plot ANOVA with 'id' is a
random effect.

### Reaction times

{% highlight r %}
# Bayes factor package requires factors, not numerical values
datff_rt <- mutate(dat_rt, truth = factor(truth), repr = factor(repr), instr = factor(instr), id = factor(id))
bf_rt <- anovaBF(RT ~ repr*truth*instr + id, whichRandom = 'id', data = datff_rt)

bf_rt
{% endhighlight %}

Let's find the model with the highest Bayes factor

{% highlight r %}
max(bf_rt)
{% endhighlight %}

### Errors

{% highlight r %}
datff <- mutate(datf, truth = factor(truth), repr = factor(repr), instr = factor(instr), id = factor(id))
bf_err <- anovaBF(err ~ repr*truth*instr + id, whichRandom = 'id', data = datff)

bf_err
{% endhighlight %}

Let's find the model with the highest Bayes factor

{% highlight r %}
max(bf_err)
{% endhighlight %}
