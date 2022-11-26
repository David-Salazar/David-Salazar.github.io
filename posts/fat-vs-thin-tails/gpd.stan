functions {
  real gpareto_lpdf(vector y, real ymin, real xi, real beta) {
    // generalised Pareto log pdf 
    int N = rows(y);
    real inv_xi = inv(xi);
    if (xi<0 && max(y-ymin)/beta > -inv_xi)
      reject("xi<0 and max(y-ymin)/beta > -1/xi; found xi, beta =", xi, beta);
    if (beta<=0)
      reject("beta<=0; found beta =", beta);
    if (fabs(xi) > 1e-15)
      return -(1+inv_xi)*sum(log1p((y-ymin) * (xi/beta))) -N*log(beta);
    else
      return -sum(y-ymin)/beta -N*log(beta); // limit xi->0
  }
  real gpareto_cdf(vector y, real ymin, real xi, real beta) {
    // generalised Pareto cdf
    real inv_xi = inv(xi);
    if (xi<0 && max(y-ymin)/beta > -inv_xi)
      reject("xi<0 and max(y-ymin)/beta > -1/xi; found xi, beta =", xi, beta);
    if (beta<=0)
      reject("beta<=0; found beta =", beta);
    if (fabs(xi) > 1e-15)
      return exp(sum(log1m_exp((-inv_xi)*(log1p((y-ymin) * (xi/beta))))));
    else
      return exp(sum(log1m_exp(-(y-ymin)/beta))); // limit xi->0
  }
  real gpareto_lcdf(vector y, real ymin, real xi, real beta) {
    // generalised Pareto log cdf
    real inv_xi = inv(xi);
    if (xi<0 && max(y-ymin)/beta > -inv_xi)
      reject("xi<0 and max(y-ymin)/beta > -1/xi; found xi, beta =", xi, beta);
    if (beta<=0)
      reject("beta<=0; found beta =", beta);
    if (fabs(xi) > 1e-15)
      return sum(log1m_exp((-inv_xi)*(log1p((y-ymin) * (xi/beta)))));
    else
      return sum(log1m_exp(-(y-ymin)/beta)); // limit xi->0
  }
  real gpareto_lccdf(vector y, real ymin, real xi, real beta) {
    // generalised Pareto log ccdf
    real inv_xi = inv(xi);
    if (xi<0 && max(y-ymin)/beta > -inv_xi)
      reject("xi<0 and max(y-ymin)/beta > -1/xi; found xi, beta =", xi, beta);
    if (beta<=0)
      reject("beta<=0; found beta =", beta);
    if (fabs(xi) > 1e-15)
      return (-inv_xi)*sum(log1p((y-ymin) * (xi/beta)));
    else
      return -sum(y-ymin)/beta; // limit xi->0
  }
  real gpareto_rng(real ymin, real xi, real beta) {
    // generalised Pareto rng
    if (beta<=0)
      reject("beta<=0; found beta =", beta);
    if (fabs(xi) > 1e-15)
      return ymin + (uniform_rng(0,1)^-xi -1) * beta / xi;
    else
      return ymin - beta*log(uniform_rng(0,1)); // limit xi->0
  }
}
data {
  real ymin;
  int<lower=0> N;
  vector<lower=ymin>[N] y;
}
transformed data {
  real ymax = max(y);
}
parameters {
  real<lower=0> beta; 
  real<lower=-beta/(ymax-ymin)> xi; 
}
model {
  y ~ gpareto(ymin, xi, beta);
  xi ~ normal(1, 1);
  beta ~ normal(1000, 300);
}
generated quantities {
  vector[N] log_lixi;
  vector[N] yrep;
  for (n in 1:N) {
    log_lixi[n] = gpareto_lpdf(rep_vector(y[n],1) | ymin, xi, beta);
    yrep[n] = gpareto_rng(ymin, xi, beta);}
}

