---
layout: post
title: "The Science behind the Magic: New paper on Harry Potter"
---



# Introduction & Preparation
We read the data in.

{% highlight r %}
library('psych')
library('broom')
library('papaja')
library('stringr')
library('reshape2')
library('magrittr')
library('tidyverse')
theme_set(theme_apa())

dat <- read.csv('harry.csv', sep = ';')
describe(select(dat, -starts_with('PVQ')), skew = FALSE)
{% endhighlight %}



{% highlight text %}
##                        vars   n   mean     sd min max range   se
## ID                        1 988 494.50 285.36   1 988   987 9.08
## Sorting_completed_YN*     2 988   1.79   0.41   1   2     1 0.01
## Sorting_house*            3 988   2.35   1.05   1   4     3 0.03
## Sorting_house_wish*       4 988   2.45   1.01   1   4     3 0.03
## age                       5 987  22.80   5.63  11 122   111 0.18
## occupation*               6 988  12.80   6.26   1  20    19 0.20
## gender*                   7 988   1.17   0.38   1   2     1 0.01
## Country_C_G_O*            8 988   2.39   0.81   1   3     2 0.03
## Bravery*                  9 988   1.04   0.28   1   4     3 0.01
## Hardwork*                10 988   2.32   0.61   1   4     3 0.02
## Intelligence*            11 988   2.89   0.57   1   4     3 0.02
## Ambition*                12 988   3.53   0.97   1   4     3 0.03
## Daring*                  13 988   1.49   1.04   1   4     3 0.03
## Dedication*              14 988   2.09   0.74   1   4     3 0.02
## Knowledge*               15 988   2.84   0.53   1   4     3 0.02
## Cunning*                 16 988   3.70   0.79   1   4     3 0.03
## Extraverted*             17 988   1.59   1.01   1   4     3 0.03
## Agreeable*               18 988   1.99   0.35   1   4     3 0.01
## Clever*                  19 988   2.89   0.70   1   4     3 0.02
## Manipulative*            20 988   3.92   0.46   1   4     3 0.01
## SD3_Machiavellianism     21 988  26.77   6.49  10  45    35 0.21
## SD3_Narcissism           22 988  23.38   5.94   9  45    36 0.19
## SD3_Psychopathy          23 988  19.25   5.59   9  45    36 0.18
## IPIP_Extraversion        24 988  28.71   8.43  10  50    40 0.27
## IPIP_Agreeableness       25 988  39.43   6.06  14  50    36 0.19
## IPIP_Conscientiousness   26 988  35.52   5.18  13  50    37 0.16
## IPIP_EmStability         27 988  29.13   7.85  10  49    39 0.25
## IPIP_Intellect           28 988  40.34   5.48  17  50    33 0.17
{% endhighlight %}

Let's sum up the individual questionnaire data to single scores, after removing people younger than 18 or older than 100.


{% highlight r %}
recode <- function(house, sorted) as.factor(as.numeric(house == sorted))
sum_over <- function(score, dat) rowSums(select(dat, one_of(paste0('PVQ_', score))))

SELF_SCORE <- c('01', '23', '39', '16', '30', '23')
STIM_SCORE <- c('10', '28', '43')
HEDON_SCORE <- c('03', '36', '46')
ACHIEVEMENT_SCORE <- c('17', '32', '48')
POWER_SCORE <- c('06', '29', '41', '12', '20', '44')
SECUR_SCORE <- c('13', '26', '53', '02', '35', '50')
CONFOR_SCORE <- c('15', '31', '42', '04', '22', '51')
TRADIT_SCORE <- c('18', '33', '40', '07', '38', '54')
BENEV_SCORE <- c('11', '25', '47', '19', '27', '55')
UNIV_SCORE <- c('08', '21', '45', '05', '37', '52', '14', '34', '57')

dat <- dat %>% 
  filter(age >= 18 & age <= 100) %>% 
  mutate(
    match = Sorting_house == Sorting_house_wish,
    Gryffindor = recode('Gryffindor', Sorting_house),
    Hufflepuff = recode('Hufflepuff', Sorting_house),
    Ravenclar  = recode('Ravenclaw', Sorting_house),
    Slytherin  = recode('Slytherin', Sorting_house),
    
    self = sum_over(SELF_SCORE, .),
    stim = sum_over(STIM_SCORE, .),
    hedon = sum_over(HEDON_SCORE, .),
    achievement = sum_over(ACHIEVEMENT_SCORE, .),
    power = sum_over(POWER_SCORE, .),
    secur = sum_over(SECUR_SCORE, .),
    confor = sum_over(CONFOR_SCORE, .),
    tradit = sum_over(TRADIT_SCORE, .),
    benev = sum_over(BENEV_SCORE, .),
    univ = sum_over(UNIV_SCORE, .)
  ) %>% 
  rename(ach = achievement) %>% 
  select(-starts_with('PVQ_'))

describe(select(dat, self:univ), skew = FALSE)
{% endhighlight %}



{% highlight text %}
##        vars   n  mean   sd min max range   se
## self      1 896 24.88 3.53   9  30    21 0.12
## stim      2 896 12.47 3.15   3  18    15 0.11
## hedon     3 896 13.70 2.65   5  18    13 0.09
## ach       4 896 13.41 2.94   4  18    14 0.10
## power     5 896 16.29 6.12   6  36    30 0.20
## secur     6 896 25.34 4.83   9  36    27 0.16
## confor    7 896 21.52 6.22   6  36    30 0.21
## tradit    8 896 19.86 5.06   6  36    30 0.17
## benev     9 896 30.18 4.23  12  36    24 0.14
## univ     10 896 41.76 7.09  14  54    40 0.24
{% endhighlight %}

## Hypotheses
Our hypotheses are as follows

- **Gryffindor** should be higher on **Extraversion** than the other houses
- **Ravenclaw** should be higher on **Intellect** than the other houses
- **Hufflepuff** should be higher on **Agreeableness and Conscientiousness** than the other houses
- **Slytherin** should be higher on **Machiavellianism, Narcissism, and Psychopathy** than the other houses

For the human value data, we do not have clear hypotheses but will still look at the data in an exploratory fashion. Our analysis plan is as follows.

But first, let's take a look at the data.

# Data visualisation
## IPIP Scores
Let's first take a look at the IPIP questionnaire data across the houses.
![plot of chunk unnamed-chunk-3](/figure/source/2017-01-22-harry-potter-analysis/unnamed-chunk-3-1.png)

Above we see that there are no big differences between the houses in general. However, on 'Agreeableness', Slytherin is reduced, while Hufflepuff has an increased value. Additionally, it seems that Gryffindor has higher extraversion than the other ones.

However, it's always good to plot more than just the mean (dough!).
![plot of chunk unnamed-chunk-4](/figure/source/2017-01-22-harry-potter-analysis/unnamed-chunk-4-1.png)

## Dark Triad
Let's look at the Dark Triad. Here the differences seem much more pronounced.


{% highlight r %}
SD_dat <- make_mean_df(starts_with('SD')) %>% 
  mutate(variable = str_replace(as.character(variable), 'SD3_', ''))
plot_mean_df(SD_dat, 'Mean Dark Triad across Houses')
{% endhighlight %}

![plot of chunk unnamed-chunk-5](/figure/source/2017-01-22-harry-potter-analysis/unnamed-chunk-5-1.png)

This is beautiful! It seems that Slytherin scores highest on all aspects of the Dark Triad, while Hufflepuff scores lowest. Exactly as one would expect.


{% highlight r %}
SD_full <- select(dat, Sorting_house, starts_with('SD3')) %>% 
  melt %>% 
  mutate(variable = str_replace(as.character(variable), 'SD3_', ''))

plot_full_df(SD_full, 'Dark Triad across Houses')
{% endhighlight %}

![plot of chunk unnamed-chunk-6](/figure/source/2017-01-22-harry-potter-analysis/unnamed-chunk-6-1.png)


## Human Values
Exploratory stuff, and not much to see here except that Slytherin is higher on power and lower on conformity.


{% highlight r %}
human_values <- make_mean_df(34:43)
plot_mean_df(human_values, 'Human values across Houses')
{% endhighlight %}

![plot of chunk unnamed-chunk-7](/figure/source/2017-01-22-harry-potter-analysis/unnamed-chunk-7-1.png)


{% highlight r %}
human_values_full <- select(dat, Sorting_house, self:univ) %>%  melt
plot_full_df(human_values_full, 'Human values across houses')
{% endhighlight %}

![plot of chunk unnamed-chunk-8](/figure/source/2017-01-22-harry-potter-analysis/unnamed-chunk-8-1.png)


{% highlight r %}
library('mgm')
library('qgraph')

dat_val <- select(dat, self:univ)

fit_graph <- function(dat, house) {
  lev <- rep(1, 10)
  type <- rep('g', 10)
  dat_val <- dat %>% 
    filter(Sorting_house == house) %>% 
    select(self:univ)
  
  mgmfit(dat_val, type = type, lev = lev, lambda.sel = 'EBIC', d = 2)
}

plot_graph <- function(fit, dat, title) {
  groups_type <- as.list(setNames(1:10, names(dat)))
  
  qgraph(
    fit$adj, 
    vsize = 6, 
    esize = 4,
    layout = "spring",
    edge.labels = round(fit$wadj, 2),
    edge.color = rgb(33, 33, 33, 100, maxColorValue = 255), 
    border.width = 1.5,
    border.color = "black",
    groups = groups_type,
    title = title,
    labels = names(groups_type),
    legend = FALSE
    )
}
{% endhighlight %}


{% highlight r %}
fit_g <- fit_graph(dat, 'Gryffindor')
{% endhighlight %}



{% highlight text %}
##   |                                                                    |                                                            |   0%  |                                                                    |------                                                      |  10%  |                                                                    |------------                                                |  20%  |                                                                    |------------------                                          |  30%  |                                                                    |------------------------                                    |  40%  |                                                                    |------------------------------                              |  50%  |                                                                    |------------------------------------                        |  60%  |                                                                    |------------------------------------------                  |  70%  |                                                                    |------------------------------------------------            |  80%  |                                                                    |------------------------------------------------------      |  90%  |                                                                    |------------------------------------------------------------| 100%
##  Note that signs of edge weights (if defined) are stored in fitobject$signs. See ?mgmfit for more info.
{% endhighlight %}



{% highlight r %}
fit_s <- fit_graph(dat, 'Slytherin')
{% endhighlight %}



{% highlight text %}
##   |                                                                    |                                                            |   0%  |                                                                    |------                                                      |  10%  |                                                                    |------------                                                |  20%  |                                                                    |------------------                                          |  30%  |                                                                    |------------------------                                    |  40%  |                                                                    |------------------------------                              |  50%  |                                                                    |------------------------------------                        |  60%  |                                                                    |------------------------------------------                  |  70%  |                                                                    |------------------------------------------------            |  80%  |                                                                    |------------------------------------------------------      |  90%  |                                                                    |------------------------------------------------------------| 100%
##  Note that signs of edge weights (if defined) are stored in fitobject$signs. See ?mgmfit for more info.
{% endhighlight %}



{% highlight r %}
fit_h <- fit_graph(dat, 'Hufflepuff')
{% endhighlight %}



{% highlight text %}
##   |                                                                    |                                                            |   0%  |                                                                    |------                                                      |  10%  |                                                                    |------------                                                |  20%  |                                                                    |------------------                                          |  30%  |                                                                    |------------------------                                    |  40%  |                                                                    |------------------------------                              |  50%  |                                                                    |------------------------------------                        |  60%  |                                                                    |------------------------------------------                  |  70%  |                                                                    |------------------------------------------------            |  80%  |                                                                    |------------------------------------------------------      |  90%  |                                                                    |------------------------------------------------------------| 100%
##  Note that signs of edge weights (if defined) are stored in fitobject$signs. See ?mgmfit for more info.
{% endhighlight %}



{% highlight r %}
fit_r <- fit_graph(dat, 'Ravenclaw')
{% endhighlight %}



{% highlight text %}
##   |                                                                    |                                                            |   0%  |                                                                    |------                                                      |  10%  |                                                                    |------------                                                |  20%  |                                                                    |------------------                                          |  30%  |                                                                    |------------------------                                    |  40%  |                                                                    |------------------------------                              |  50%  |                                                                    |------------------------------------                        |  60%  |                                                                    |------------------------------------------                  |  70%  |                                                                    |------------------------------------------------            |  80%  |                                                                    |------------------------------------------------------      |  90%  |                                                                    |------------------------------------------------------------| 100%
##  Note that signs of edge weights (if defined) are stored in fitobject$signs. See ?mgmfit for more info.
{% endhighlight %}

![plot of chunk unnamed-chunk-11](/figure/source/2017-01-22-harry-potter-analysis/unnamed-chunk-11-1.png)![plot of chunk unnamed-chunk-11](/figure/source/2017-01-22-harry-potter-analysis/unnamed-chunk-11-2.png)![plot of chunk unnamed-chunk-11](/figure/source/2017-01-22-harry-potter-analysis/unnamed-chunk-11-3.png)![plot of chunk unnamed-chunk-11](/figure/source/2017-01-22-harry-potter-analysis/unnamed-chunk-11-4.png)


# Data Analysis
## Bayesian analysis
Here I distinguish between *model comparison*, and *parameter estimation*. ROPE butchers the distinction because it thinks it can use parameter estimation to do model comparison.

Our hypothesis is that Slytherin has a higher machiavellianism score than the average of the other three houses. Therefore, we see how often the mean of Slytherin is higher than the mean of the other three houses (we can, of course, test more interesting hypotheses). For more, see [here](http://bayesfactor.blogspot.co.at/2015/01/multiple-comparisons-with-bayesfactor-2.html).

The Bayes factor is

$$
\begin{align*}
\text{posterior odds} =& \text{prior odds} \times \text{Bayes factor} \\
\text{Bayes factor} =& \frac{\text{posterior odds}}{\text{prior odds}}
\end{align*}
$$

We want to test our order restricted hypothesis against the null hypothesis. Luckily, the Bayes factor is transitive, such that

$$
\begin{align*}
\frac{p(\textbf{y}|M_r)}{p(\textbf{y}|M_0)} = \frac{p(\textbf{y}|M_r)}{p(\textbf{y}|M_f)} \cdot \frac{p(\textbf{y}|M_f)}{p(\textbf{y}|M_0)} 
\end{align*}
$$

where $M_r$ denotes the restricted model in which $\text{Slytherin} > \text{Mean(Gryffindor, Hufflepuff, Ravenclar)}$, $M_0$ denotes the null model in which $\text{Gryffindor} = \text{Slytherin} = \text{Hufflepuff} = \text{Ravenclaw}$, and the full model has a parameter for each mean, such that they can vary independently without constraint. So we can compute the Bayes factor we are interested in as follows.

Woho! Huge Bayes factor that the means are different. Well, that's rather obvious and not of interest, though.

{% highlight r %}
library('BayesFactor')

(bf_full_zero <- anovaBF(SD3_Machiavellianism ~ Sorting_house, data = dat))
{% endhighlight %}



{% highlight text %}
## Bayes factor analysis
## --------------
## [1] Sorting_house : 1.118886e+18 ±0%
## 
## Against denominator:
##   Intercept only 
## ---
## Bayes factor type: BFlinearModel, JZS
{% endhighlight %}

Now let's estimate the posterior means and variance (we do not model heteroscedasticity; seems there is none anyway). This is is sum-to-zero coded.

{% highlight r %}
post <- posterior(bf_full_zero, iterations = 10000)
head(post)
{% endhighlight %}



{% highlight text %}
## Markov Chain Monte Carlo (MCMC) output:
## Start = 1 
## End = 7 
## Thinning interval = 1 
##            mu Sorting_house-Gryffindor Sorting_house-Hufflepuff
## [1,] 27.03187               -0.5003774                -2.645128
## [2,] 26.83707                0.0213525                -3.100425
## [3,] 26.67950               -0.6162571                -2.479251
## [4,] 27.06214               -0.8950938                -2.392838
## [5,] 27.50802               -0.5991079                -2.779407
## [6,] 26.84725               -0.7558496                -2.420132
## [7,] 27.20927               -0.5565723                -1.561983
##      Sorting_house-Ravenclaw Sorting_house-Slytherin     sig2
## [1,]              -0.5694078                3.714914 38.92887
## [2,]              -0.4017143                3.480786 34.87370
## [3,]              -1.0521504                4.147658 37.61376
## [4,]              -0.6025914                3.890524 41.56375
## [5,]              -0.1566038                3.535119 37.98946
## [6,]              -0.3082137                3.484195 36.24532
## [7,]              -1.0714255                3.189981 36.76806
##      g_Sorting_house
## [1,]      0.08214204
## [2,]      0.22175841
## [3,]      0.32424025
## [4,]      0.06324983
## [5,]      0.17019969
## [6,]      0.97428578
## [7,]      0.13792799
{% endhighlight %}


{% highlight r %}
main_house <- 'Slytherin'
other_houses <- c('Gryffindor', 'Hufflepuff', 'Ravenclaw')

# for the simple Helmert contrast
bayes_factor_restr_zero <- function(bf_full_zero, main_house, other_houses, iterations = 100000) {
  
  post <- posterior(bf_full_zero, iterations = iterations)
  mu <- post[, 1]
  header <- colnames(post)
  
  # compute the mean of the house we want to test the others against
  main <- mu + post[, grepl(main_house, header)]
  
  # compute the mean of the other three houses
  other <- sapply(other_houses, function(c) {
    mu + post[, grepl(c, header)]
    }) %>% rowMeans
  
  prior_odds <- 1/2
  posterior_odds <- mean(main > other) / nrow(post)
  
  bf_restr_full <- posterior_odds / prior_odds
  bf_restr_zero <- bf_restr_full * extractBF(bf_full_zero)$bf
  
  log(bf_restr_zero)
}

bayes_factor_restr_zero(bf_full_zero, main_house, other_houses)
{% endhighlight %}



{% highlight text %}
## [1] 30.73909
{% endhighlight %}

However, using this method, we can test more interesting constraints. For example, we might be interested in orderings, say

$$
H_1: Gryffindor > Slytherin, Hufflepuff, Ravenclaw
$$

# Appendix
## Frequentist Analysis
The authors used the following Helmert contrast for their linear models.


{% highlight r %}
helmert <- matrix(c(3/4, -1/4, -1/4, -1/4, 0, 2/3, -1/3, -1/3, 0, 0, 1/2, -1/2), ncol = 3)
helmert
{% endhighlight %}



{% highlight text %}
##       [,1]       [,2] [,3]
## [1,]  0.75  0.0000000  0.0
## [2,] -0.25  0.6666667  0.0
## [3,] -0.25 -0.3333333  0.5
## [4,] -0.25 -0.3333333 -0.5
{% endhighlight %}

As an example, let's run a model

{% highlight r %}
m <- lm(dat$IPIP_Intellect ~ Sorting_house, data = dat)
tidy(m)
{% endhighlight %}



{% highlight text %}
##                      term   estimate std.error  statistic     p.value
## 1             (Intercept) 40.6356275 0.3446314 117.910414 0.000000000
## 2 Sorting_houseHufflepuff -1.8294601 0.4980020  -3.673600 0.000253405
## 3  Sorting_houseRavenclaw  0.9203291 0.4740024   1.941613 0.052498583
## 4  Sorting_houseSlytherin -0.6287310 0.5666486  -1.109561 0.267487462
{% endhighlight %}

Therefore, the effects denote:

- **Sorting_house1**: Gryffindor against Hufflepuff + Ravenclaw + Slytherin
- **Sorting_house2**: Hufflepuff against Ravenclaw + Slytherin
- **Sorting_house3**: Ravenclaw against Slytherin

However, for the *Intellect* response, this is not what we want, since we want to compare Ravenclaw against all others.

{% highlight r %}
dat <- mutate(dat, Sorting_house = relevel(Sorting_house, ref = 'Ravenclaw'))
mclaw <- lm(dat$IPIP_Intellect ~ Sorting_house, data = dat)
tidy(mclaw)
{% endhighlight %}



{% highlight text %}
##                      term   estimate std.error  statistic
## 1             (Intercept) 41.5559567 0.3254344 127.693822
## 2 Sorting_houseGryffindor -0.9203291 0.4740024  -1.941613
## 3 Sorting_houseHufflepuff -2.7497893 0.4849151  -5.670661
## 4  Sorting_houseSlytherin -1.5490601 0.5551823  -2.790183
##        p.value
## 1 0.000000e+00
## 2 5.249858e-02
## 3 1.921122e-08
## 4 5.380177e-03
{% endhighlight %}


