library(dplyr)

# Function to calculate recombination rate (cM/Mb) between adjacent markers
# that are at least 2 kb apart in a given recombination map DataFrame
calc_cM_per_bp <- function(df) {
  first_row <- 1
  n <- nrow(df)
  results <- list()
  j <- 0
  
  while (j < n & first_row < n) {
    for (j in (first_row + 1):n) {
      if (df$V4[j] - df$V4[first_row] >= 2000) {
        rate <- (df$V3[j] - df$V3[first_row]) / ((df$V4[j] - df$V4[first_row]) / 1e6)
        # Add genetic start and end positions (V3)
        results[[length(results) + 1]] <- c(
          rate,
          first_row,
          j,
          df$V4[first_row],  # start_bp
          df$V4[j],          # end_bp
          df$V3[first_row],  # start_cM
          df$V3[j]           # end_cM
        )
        first_row <- j
        break
      }
    }
  }
  
  res.df <- results %>%
    unlist() %>%
    matrix(byrow = TRUE, ncol = 7) %>%
    as.data.frame()
  
  colnames(res.df) <- c("rate", "first_marker", "last_marker", "first_pos", "last_pos", "start_cM", "end_cM")
  res.df$center_bp <- (res.df$first_pos + res.df$last_pos) / 2
  
  return(res.df)
}

# Function to read a map file for a given chromosome
read_map_file <- function(chr) {
  str = paste0("decode2019.chrchr", chr, ".GRCh38.map")
  df <- read.table(str)
  return(df)
}

# Read recombination maps for chromosomes 1–22 into a list
map_list <- lapply(1:22, read_map_file)
names(map_list) <- as.character(1:22)

# Compute recombination rates for each chromosome
res_list <- lapply(map_list, calc_cM_per_bp)

# Combine results from all chromosomes into one data frame
all_res <- bind_rows(res_list, .id = "chr")

# ---- Calculate genome-wide average recombination rate ----

# Calculate genome-wide average recombination rate using chromosome-end differences
genetic_lengths <- sapply(map_list, function(df) max(df$V3) - min(df$V3))
physical_lengths <- sapply(map_list, function(df) max(df$V4) - min(df$V4))

genomic_avg_rate <- sum(genetic_lengths) / (sum(physical_lengths) / 1e6)  # cM/Mb
print(genomic_avg_rate)

# ---- Classify intervals as hotspots ----

# Mark intervals as hotspots if rate ≥ 10 × genome-wide average rate
all_res <- all_res %>%
  mutate(hotspot = rate >= 10 * genomic_avg_rate)

# Save
write.csv(all_res, "../genomewide_hotspots.csv", row.names = FALSE)
