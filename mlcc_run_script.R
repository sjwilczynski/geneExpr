library(varclust)
set.seed(42)

args        <- commandArgs(trailingOnly = TRUE)
max.dim     <- as.numeric(args[1]) # maximal subspace dimension
numb.cores  <- as.numeric(args[2]) # number of processor's cores to use
numb.runs   <- as.numeric(args[3]) # number of random initializations
max.iter    <- as.numeric(args[4]) # number of iterations of the algorithm
clusts      <- eval(parse(text=args[5])) #number of clusters to test


microarray <- as.matrix(read.csv(file="./data/microarray_train.csv", header=TRUE, sep=","))

for(num_clust in clusts){
  start.time <- Sys.time()
  print(paste("Starting mlcc.reps for", num_clust, "clusters", sep = " "))
  res <- mlcc.reps(microarray, numb.clusters = num_clust, numb.runs = numb.runs, 
                   max.dim = max.dim, numb.cores = numb.cores, max.iter = max.iter)
  print(paste("Finished mlcc.reps for", num_clust, "clusters", sep = " "))
  print("Saving results")
  filename = paste("mlcc_results/output", paste(num_clust, max.dim, numb.runs, max.iter, sep="_"), ".RData", sep = "")
  save(res, file = filename)
  #for saving a bit of memory 
  rm(res)
  end.time <- Sys.time()
  print(difftime(end.time, start.time, units = "mins"))
}