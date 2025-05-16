### This file simulates gene conversion tracts without linkage disequilibrium. 

library(dplyr)
library(data.table)
library(purrr)
library(parallel)
source("model.R")

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
  
  return(list(l_lst, psi_lst))
}

res.geom <- mclapply(1:300, do_one_geom, psi, length_chrom, mc.cores = 15)

saveRDS(res.geom, file = "res.geom.rds")