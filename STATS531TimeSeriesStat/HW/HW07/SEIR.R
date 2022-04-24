library(tidyverse)
library(pomp)

sir_step <- Csnippet("
  double dN_SE = rbinom(S,1-exp(-Beta*I/N*dt));
  double dN_EI = rbinom(E,1-exp(-mu_EI*dt));
  double dN_IR = rbinom(I,1-exp(-mu_IR*dt));
  S -= dN_SE;
  E += dN_SE - dN_EI;
  I += dN_EI - dN_IR;
  R += dN_IR;
  H += dN_IR;
")

sir_rinit <- Csnippet("
  S = nearbyint(eta*N);
  E = 0;
  I = 1;
  R = nearbyint((1-eta)*N);
  H = 0;
")

sir_dmeas <- Csnippet("
  lik = dnbinom_mu(reports,k,rho*H,give_log);
")

sir_rmeas <- Csnippet("
  reports = rnbinom_mu(k,rho*H);
")
read_csv(paste0("./Measles_Consett_1948.csv")) %>%
  select(week,reports=cases) %>% filter(week<=42) -> meas

meas %>%
  pomp(
    times="week",t0=0,
    rprocess=euler(sir_step,delta.t=1/7),
    rinit=sir_rinit,
    rmeasure=sir_rmeas,
    dmeasure=sir_dmeas,
    accumvars="H",
    statenames=c("S","E","I","R","H"),
    partrans=parameter_trans(
      log=c("Beta", "mu_EI", "mu_IR", "k"),
      logit=c("eta", "rho")
    ),
    paramnames=c("N","Beta","mu_IR", "eta", "rho", "k", "mu_EI"),
    params=c(Beta=40, mu_IR=0.35, rho=0.5, k=500, eta=0.03, N=38000, mu_EI=1.3)
  ) -> measSEIR


