###### simulating tracts ######

sim_tracts <- function(N_gene_conv, phi, length_chrom, dist_N, size = 3, max = 599) {
  start <- sample(1:length_chrom, N_gene_conv, replace = TRUE)
  if (dist_N == "geom") {lengths <- rgeom(N_gene_conv, phi) + 1}
  if (dist_N == "geom2") {lengths <- rgeom(N_gene_conv, phi) + rgeom(N_gene_conv, phi) + 2}
  if (dist_N == "nbinom") {lengths <- rnbinom(N_gene_conv, size = size, mu = phi)}
  if (dist_N == "unif") {lengths <- sample(1:max, size = N_gene_conv, replace = TRUE)}
  end <- start + lengths - 1
  start_end <- cbind(start, end)
  start_end <- start_end[start_end[,2] < length_chrom,]
  tracts <- split(start_end, seq(nrow(start_end)))
  return(tracts)
}

sim_gene_conv <- function(actual, psi) {
  psi_tr <- psi[actual[1]:actual[2]]
  changes <- rbinom(n = actual[2] - actual[1] + 1, size = 1, prob = psi_tr)
  min_pos <- min(which(changes == 1))
  max_pos <- max(which(changes == 1))
  if (min_pos == Inf) {
    return(0)
  }
  return(c(min_pos + actual[1] - 1, max_pos + actual[1] - 1))
}

###### pmf ######

pL_geom_2M <- function(l, psi, phi, M) {
  denom <- (1-phi)-(1-phi)^M
  num <- phi*(1-phi)^(l-1)
  return(num/denom)
}

pL_geom2_2M <- function(l, psi, phi, M) {
  C <- phi + psi - phi*psi
  denom <- C*( (3-M)*phi*(1-phi)^(M-1) - (1-phi)^(M-1) - 2*phi + 1 ) + 2*phi*(1-(1-phi)^(M-1))
  num <- C*(l-3)*phi^2*(1-phi)^(l-2) + 2*phi^2*(1-phi)^(l-2)
  return(num/denom)
}

neg_log_lik <- function(phi, psi_lst, pL, l_lst, M = FALSE) {
  if (M == FALSE) {
    val <- -sum(log(unlist(map2(l_lst, psi_lst, pL, phi = phi))))
    return(val)
  }
  val <- -sum(log(unlist(map2(l_lst, psi_lst, pL, phi = phi, M = M))))
  return(val)
}

# pL_geom <- function(l, psi, phi) {
#   if (l == 1) {return(phi/(phi + psi - phi*psi))}
#   else {return(phi*(1-phi)^(l-1)*psi/(phi + psi - phi*psi))}
# }
# 
# pL_geom_M <- function(l, psi, phi, M) {
#   if (l == 1) {return(phi*psi/(phi*psi + psi^2*(1 - phi - (1 - phi)^M)))}
#   else {return(phi*(1-phi)^(l-1)*psi^2/(phi*psi + psi^2*(1 - phi - (1 - phi)^M)))}
# }
# 
# pL_geom_1M <- function(l, psi, phi, M) {
#   pL_geom_M(l, psi, phi, M)/(1 - pL_geom_M(1, psi, phi, M))
# }
# 
# C <- function(phi, psi) {
#   phi + psi - phi*psi
# }
# 
# pL_geom2 <- function(l, psi, phi) {
#   if (l == 1) {return((2*phi^2*psi*(1-psi))/(C(phi,psi)*(C(phi,psi)^2 - phi^2*(1-psi)^2)))}
#   else {return((phi^2*(1-phi)^(l-2)*psi^2*((l-3)*C(phi,psi)+2))/(C(phi,psi)*(C(phi,psi)^2 - phi^2*(1-psi)^2)))}
# }
# 
# denom <- function(phi, psi, M) {
#   psi*(psi*(1-phi)^M*(phi*((M-3)*C(phi, psi)+3)+psi-phi*psi)-(phi-1)*((4*psi-2)*phi^2+psi*phi*(2*psi-2*phi*psi-3)+psi*(phi*psi-psi)))/(phi-1)
# }
# 
# pL_geom2_M <- function(l, psi, phi, M) {
#   if (l == 1) {return(2*phi^2*psi*(1-psi)/denom(phi,psi,M))}
#   else {return(phi^2*(1-phi)^(l-2)*psi^2*( (l-3)*C(phi, psi) + 2 )/denom(phi,psi,M))}
# }
# 
# pL_geom2_1M <- function(l, psi, phi, M) {
#   pL_geom2_M(l, psi, phi, M)/(1 - pL_geom2_M(1, psi, phi, M))
# }
# 
# 
# get_quantile_geom <- function(psi, phi, p) {
#   i <- 1
#   F_geom <- pL_geom(1, psi, phi)
#   while(F_geom < p) {
#     i <- i + 1
#     F_geom <- F_geom + pL_geom(i, psi, phi)
#   }
#   return(i)
# }
# 
# get_quantile_geom2 <- function(psi, phi, p) {
#   i <- 1
#   F_geom2 <- pL_geom2(1, psi, phi)
#   while(F_geom2 < p) {
#     i <- i + 1
#     F_geom2 <- F_geom2 + pL_geom2(i, psi, phi)
#   }
#   return(i)
# }

###### calculating psi ######

est_psi <- function(observed, psi, region, length_chrom, debias) {
  leftmost <- max(observed[1] - region, 1)
  rightmost <- min(observed[2] + region, length_chrom)
  index <- leftmost:rightmost
  if (debias == TRUE) {
    if (observed[1] != observed[2]) {
      index <- setdiff(index, observed[1])
    }
  }
  return(mean(psi[index]))
}
