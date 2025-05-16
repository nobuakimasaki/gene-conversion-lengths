This directory contains the code and data used to run the simulation in the Appendix of the main paper. In this simulation, tracts are simulated on individuals generated from the first seed of the coalescent simulation using msprime. We later fit our model on the simulated tracts.

The "data" directory contains the .vcf files representing the genotypes of individuals generated using msprime, and the minor allele frequencies of the generated region.

The "sim_tracts" directory contains python files used to simulate observed tracts using the .vcf files. They also contain the tracts simulated from each distribution of the gene conversion tract lengths.

The "model_fitting" directory contains the R code used to fit the model on the simulated observed tracts. They also contain results of fitting the model.

"analysis.Rmd" is the file used to analyze the estimates obtained from fitting our model on the simulated observed tracts.