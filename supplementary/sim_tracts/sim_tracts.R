### This file simulates gene conversion tracts without linkage disequilibrium. 
### Additionally, we fit the model to the generated tracts, but this part of the code is not used in the Supplementary Materials
### Furthermore, we only use res.geom.rds in Supplementary Materials

library(dplyr)
library(data.table)
library(purrr)
library(parallel)
source("fns_sim.R")

boot_MLE <- function(x, l_lst, psi_lst, M) {
  ind <- sample(1:length(l_lst), replace = TRUE)
  l_lst_boot <- l_lst[ind]
  psi_lst_boot <- psi_lst[ind]
  
  optim.out.geom <- optim(0.005, neg_log_lik, psi_lst = psi_lst_boot, pL = pL_geom_2M,
                          l_lst = l_lst_boot, M = M, lower = 0.0001, upper = 0.05, method = "Brent")
  optim.out.geom2 <- optim(0.005, neg_log_lik, psi_lst = psi_lst_boot, pL = pL_geom2_2M,
                          l_lst = l_lst_boot, M = M, lower = 0.0001, upper = 0.05, method = "Brent")
  
  print("finished 1 bootstrap iteration")
  return(c(optim.out.geom$par, optim.out.geom2$par))
}

# Read in Chromosome 1 from UK Biobank whole autosome data
MAF.df <- read.table("/projects/browning/brwnlab/sharon/for_nobu/gc_length/ukbiobank/chr1.allregions.pmaf.gz")
print(head(MAF.df))

# Add 1 to position index
MAF.df$pos <- MAF.df$V2 + 1
length_chrom <- max(MAF.df$pos) + 100
psi <- numeric(length_chrom)
# Define a vector of heterozygosity probabilities for each position
psi[c(MAF.df$pos)] <- 2*MAF.df$V3*(1-MAF.df$V3)

# Remove markers with MAF smaller than 0.05
exc <- MAF.df[MAF.df$V3 < 0.05,]
psi[exc$pos] <- 0

do_one_geom <- function(i, psi, length_chrom) {
  print(paste0("iteration: ", i))
  
  actual_trts <- sim_tracts(10^5, 1/300, length_chrom, "geom")
  obs_trts <- suppressWarnings(lapply(actual_trts, sim_gene_conv, psi))
  
  # Remove 0s or tracts greater than 1500
  remove <- c()
  for (i in 1:length(obs_trts)) {
    if (obs_trts[[i]][1] == 0) {
      remove <- c(remove, i)
    }
  }
  obs_trts <- obs_trts[-remove]
  
  # Draw L from indices
  l_lst <- lapply(obs_trts, function(x) {x[2] - x[1] + 1}) %>% unlist()
  m <- length(l_lst)
  print(paste0("m: ", m))
  m1 <- sum(unlist(l_lst) == 1)
  print(paste0("m1: ", m1))
  
  # Calculate psi_j
  psi_lst <- lapply(obs_trts, est_psi, psi, 5000, length_chrom, FALSE) %>% unlist()
  
  # Remove tracts with L == 1 or L > 1500
  remove1 <- which(l_lst == 1 | l_lst > 1500)
  l_lst_filt <- l_lst[-remove1]
  psi_lst_filt <- psi_lst[-remove1]
  
  optim.out.geom <- optim(0.005, neg_log_lik, psi_lst = psi_lst_filt, pL = pL_geom_2M,
                          l_lst = l_lst_filt, M = 1500, lower = 0.0001, upper = 0.05, method = "Brent")
  optim.out.geom2 <- optim(0.005, neg_log_lik, psi_lst = psi_lst_filt, pL = pL_geom2_2M,
                          l_lst = l_lst_filt, M = 1500, lower = 0.0001, upper = 0.05, method = "Brent")
  
  boot.res <- lapply(1:500, boot_MLE, l_lst_filt, psi_lst_filt, M = 1500)
  
  return(list(optim.out.geom, optim.out.geom2, boot.res, l_lst, psi_lst))
}

do_one_geom2 <- function(i, psi, length_chrom) {
  print(paste0("iteration: ", i))
  
  actual_trts <- sim_tracts(10^5, 1/150, length_chrom, "geom2")
  obs_trts <- suppressWarnings(lapply(actual_trts, sim_gene_conv, psi))
  
  # Remove 0s or tracts greater than 1500
  remove <- c()
  for (i in 1:length(obs_trts)) {
    if (obs_trts[[i]][1] == 0) {
      remove <- c(remove, i)
    }
  }
  obs_trts <- obs_trts[-remove]
  
  # Draw L from indices
  l_lst <- lapply(obs_trts, function(x) {x[2] - x[1] + 1}) %>% unlist()
  m <- length(l_lst)
  print(paste0("m: ", m))
  m1 <- sum(unlist(l_lst) == 1)
  print(paste0("m1: ", m1))
  
  # Calculate psi_j
  psi_lst <- lapply(obs_trts, est_psi, psi, 5000, length_chrom, FALSE) %>% unlist()
  
  # Remove tracts with L == 1 or L > 1500
  remove1 <- which(l_lst == 1 | l_lst > 1500)
  l_lst_filt <- l_lst[-remove1]
  psi_lst_filt <- psi_lst[-remove1]
  
  optim.out.geom <- optim(0.005, neg_log_lik, psi_lst = psi_lst_filt, pL = pL_geom_2M,
                          l_lst = l_lst_filt, M = 1500, lower = 0.0001, upper = 0.05, method = "Brent")
  optim.out.geom2 <- optim(0.005, neg_log_lik, psi_lst = psi_lst_filt, pL = pL_geom2_2M,
                           l_lst = l_lst_filt, M = 1500, lower = 0.0001, upper = 0.05, method = "Brent")
  
  boot.res <- lapply(1:500, boot_MLE, l_lst_filt, psi_lst_filt, M = 1500)
  
  return(list(optim.out.geom, optim.out.geom2, boot.res, l_lst, psi_lst))
}

do_one_nbinom <- function(i, psi, length_chrom) {
  print(paste0("iteration: ", i))
  
  actual_trts <- sim_tracts(10^5, 300, length_chrom, "nbinom", size = 3)
  obs_trts <- suppressWarnings(lapply(actual_trts, sim_gene_conv, psi))
  
  # Remove 0s or tracts greater than 1500
  remove <- c()
  for (i in 1:length(obs_trts)) {
    if (obs_trts[[i]][1] == 0) {
      remove <- c(remove, i)
    }
  }
  obs_trts <- obs_trts[-remove]
  
  # Draw L from indices
  l_lst <- lapply(obs_trts, function(x) {x[2] - x[1] + 1}) %>% unlist()
  m <- length(l_lst)
  print(paste0("m: ", m))
  m1 <- sum(unlist(l_lst) == 1)
  print(paste0("m1: ", m1))
  
  # Calculate psi_j
  psi_lst <- lapply(obs_trts, est_psi, psi, 5000, length_chrom, FALSE) %>% unlist()
  
  # Remove tracts with L == 1 or L > 1500
  remove1 <- which(l_lst == 1 | l_lst > 1500)
  l_lst_filt <- l_lst[-remove1]
  psi_lst_filt <- psi_lst[-remove1]
  
  optim.out.geom <- optim(0.005, neg_log_lik, psi_lst = psi_lst_filt, pL = pL_geom_2M,
                          l_lst = l_lst_filt, M = 1500, lower = 0.0001, upper = 0.05, method = "Brent")
  optim.out.geom2 <- optim(0.005, neg_log_lik, psi_lst = psi_lst_filt, pL = pL_geom2_2M,
                           l_lst = l_lst_filt, M = 1500, lower = 0.0001, upper = 0.05, method = "Brent")
  
  boot.res <- lapply(1:500, boot_MLE, l_lst_filt, psi_lst_filt, M = 1500)
  
  return(list(optim.out.geom, optim.out.geom2, boot.res, l_lst, psi_lst))
}

do_one_unif <- function(i, psi, length_chrom) {
  print(paste0("iteration: ", i))
  
  actual_trts <- sim_tracts(10^5, phi = 1, length_chrom = length_chrom, dist_N = "unif", max = 599)
  obs_trts <- suppressWarnings(lapply(actual_trts, sim_gene_conv, psi))
  
  # Remove 0s or tracts greater than 1500
  remove <- c()
  for (i in 1:length(obs_trts)) {
    if (obs_trts[[i]][1] == 0) {
      remove <- c(remove, i)
    }
  }
  obs_trts <- obs_trts[-remove]
  
  # Draw L from indices
  l_lst <- lapply(obs_trts, function(x) {x[2] - x[1] + 1}) %>% unlist()
  m <- length(l_lst)
  print(paste0("m: ", m))
  m1 <- sum(unlist(l_lst) == 1)
  print(paste0("m1: ", m1))
  
  # Calculate psi_j
  psi_lst <- lapply(obs_trts, est_psi, psi, 5000, length_chrom, FALSE) %>% unlist()
  
  # Remove tracts with L == 1 or L > 1500
  remove1 <- which(l_lst == 1 | l_lst > 1500)
  l_lst_filt <- l_lst[-remove1]
  psi_lst_filt <- psi_lst[-remove1]
  
  optim.out.geom <- optim(0.005, neg_log_lik, psi_lst = psi_lst_filt, pL = pL_geom_2M,
                          l_lst = l_lst_filt, M = 1500, lower = 0.0001, upper = 0.05, method = "Brent")
  optim.out.geom2 <- optim(0.005, neg_log_lik, psi_lst = psi_lst_filt, pL = pL_geom2_2M,
                           l_lst = l_lst_filt, M = 1500, lower = 0.0001, upper = 0.05, method = "Brent")
  
  boot.res <- lapply(1:500, boot_MLE, l_lst_filt, psi_lst_filt, M = 1500)
  
  return(list(optim.out.geom, optim.out.geom2, boot.res, l_lst, psi_lst))
}

res.geom <- mclapply(1:300, do_one_geom, psi, length_chrom, mc.cores = 15)
res.geom2 <- mclapply(1:300, do_one_geom2, psi, length_chrom, mc.cores = 15)
res.nbinom <- mclapply(1:300, do_one_nbinom, psi, length_chrom, mc.cores = 15)
res.unif <- mclapply(1:300, do_one_unif, psi, length_chrom, mc.cores = 15)

saveRDS(res.geom, file = "res.geom.rds")
saveRDS(res.geom2, file = "res.geom2.rds")
saveRDS(res.nbinom, file = "res.nbinom.rds")
saveRDS(res.unif, file = "res.unif.rds")