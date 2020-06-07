
data {
  int<lower=1> N_north;
  int<lower=1> N_south;
  real x_north[N_north];
  real x_south[N_south];

  real<lower=0> rho_north;
  real<lower=0> rho_south;
  real<lower=0> eta_north;
  real<lower=0> eta_south;
}

transformed data {
  matrix[N_north, N_north] cov_north = cov_exp_quad(x_north, eta_north, rho_north)
                     + diag_matrix(rep_vector(1e-10, N_north));
  matrix[N_north, N_north] L_cov_north = cholesky_decompose(cov_north);
  
  matrix[N_south, N_south] cov_south = cov_exp_quad(x_south, eta_south, rho_south)
                     + diag_matrix(rep_vector(1e-10, N_south));
                     
  matrix[N_south, N_south] L_cov_south = cholesky_decompose(cov_south);
  
}

parameters {}
model {}

generated quantities {
  vector[N_north] f_north = multi_normal_cholesky_rng(rep_vector(0.5, N_north), L_cov_north);
  vector[N_north] y_north;
  
  vector[N_south] f_south = multi_normal_cholesky_rng(rep_vector(-0.5, N_south), L_cov_south);
  vector[N_south] y_south;
  
  for (n in 1:N_north) {
    y_north[n] = bernoulli_logit_rng(f_north[n]);
  }
  
  for (n in 1:N_north) {
    y_south[n] = bernoulli_logit_rng(f_south[n]);
  }
  
}

