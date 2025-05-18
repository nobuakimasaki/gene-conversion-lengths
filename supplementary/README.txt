This directory contains the code and data used in the Supplementary Materials.

The sim_tracts directory contains "sim_tracts.R", which uses the minor allele frequencies from Chromosome 1 of the UK Biobank sequence data to simulate observed tracts. We are unable to include these minor allele frequencies in this public repository. We save the simulated observed tracts in "res.geom.rds".

The l_psi directory contains "calculate_l_psi_sim.py", which is used to calculate the observed tract lengths and estimated allele conversion probabilities from the coalescent simulation. We save these values in "l_psi_sim.csv".

"analysis.Rmd" is used to compare the observed tract lengths from these two simulations with the CDF of our model. It is used to create the related supplementary figures.