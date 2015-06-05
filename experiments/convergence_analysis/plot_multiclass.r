if(system(command="uname", intern=T) == "Darwin") library(coda)
if(system(command="uname", intern=T) == "Linux") library(coda, lib.loc="~/RLIB")


DIR_RESULTS = "multiclass/"

pdf.options(width=20,height=10,pointsize=32)

## N_VECTOR = c(5, 10, 20, 50, 100)
N_SEEDS = 10


mygelmanplot_slowest= function(mcmc_runs, nx = 100) {

  x = as.integer(seq(from=100, to=dim(mcmc_runs[[1]])[1], length.out = nx))
  slowest = which.max(gelman.diag(mcmc_runs, autoburnin=F)[[1]][,1])

  cat("Slowest variable", slowest, "\n")
  
  y = rep(0, nx)

  for(i in 1:nx) {
    y[i] = gelman.diag(window(mcmc_runs, 1, x[i]), autoburnin=F)[[1]][slowest,1]
  }

  list(x, y, slowest)
}


mygelmanplot= function(mcmc_runs, nx = 100, nout=20) {

  x = as.integer(seq(from=100, to=dim(mcmc_runs[[1]])[1], length.out = nx))
  slowest = order(gelman.diag(mcmc_runs, autoburnin=F)[[1]][,1], decreasing=TRUE)[1:nout]
  
  y = matrix(0, nx, nout)

  for(i in 1:nx) {
    y[i,] = gelman.diag(window(mcmc_runs, 1, x[i]), autoburnin=F)[[1]][slowest,1]
  }

  list(x, y)
}


OPTION_NZ = 50

for(OPTION_ARD in c("True", "False")) {

print("Plotting PSRF")
pdf(paste(DIR_RESULTS, "PLOT_convergence_multiclass_nz", OPTION_NZ, "_50_ard", OPTION_ARD, ".pdf", sep=""))

cat("ARD", OPTION_ARD, "  NZ", OPTION_NZ, "\n")

psrf = list()
ess = list()
times = list()
for(OPTION_SAMPLER in c(1:2)) {

  mcmc_runs = mcmc.list()
  ess[[OPTION_SAMPLER]] = rep(0, N_SEEDS)
  times[[OPTION_SAMPLER]] = rep(0, N_SEEDS)

  index_seed = 1
  for(OPTION_SEED in c(0:(N_SEEDS-1))) {
    filename_samples = paste(DIR_RESULTS, "SAMPLES_NZ_", OPTION_NZ, "_ARD_", OPTION_ARD, "_", c("HMC", "AA")[OPTION_SAMPLER], "_SEED_", OPTION_SEED, ".txt", sep="")
    if(file.exists(filename_samples)) aa = as.matrix(read.table(filename_samples, colClasses="numeric"))

    filename_times = paste(DIR_RESULTS, "TIME_NZ_", OPTION_NZ, "_ARD_", OPTION_ARD, "_", c("HMC", "AA")[OPTION_SAMPLER], "_SEED_", OPTION_SEED, ".txt", sep="")
    if(file.exists(filename_times)) bb = mean(as.matrix(read.table(filename_times, colClasses="numeric")))
    
    filename_accepts = paste(DIR_RESULTS, "ACCEPTS_NZ_", OPTION_NZ, "_ARD_", OPTION_ARD, "_", c("HMC", "AA")[OPTION_SAMPLER], "_SEED_", OPTION_SEED, ".txt", sep="")
    if(file.exists(filename_accepts)) cc = as.matrix(read.table(filename_accepts, colClasses="numeric"))
    
    if(min(cc) > 0.001) {
        mcmc_runs[[index_seed]] = mcmc(aa)

        ess[[OPTION_SAMPLER]][index_seed] = min(effectiveSize(aa))
        times[[OPTION_SAMPLER]][index_seed] = bb
        
        index_seed = index_seed + 1
    }
  }

  ess[[OPTION_SAMPLER]] = ess[[OPTION_SAMPLER]][1:(index_seed-1)]
  times[[OPTION_SAMPLER]] = times[[OPTION_SAMPLER]][1:(index_seed-1)]
  
  psrf[[OPTION_SAMPLER]] = mygelmanplot(mcmc_runs)
  
}

plot(psrf[[1]][[1]], psrf[[1]][[2]][,1], type="l", ylim=c(1,4), xlim=range(psrf[[1]][[1]]), xlab="Iteration", ylab="PSRF", main=paste("Multiclass dataset -", c("RBF", "ARD")[(OPTION_ARD=="True")+1], "-", OPTION_NZ, "inducing points"), col=rgb(0,0,1,0.3))

for(i in 2:(dim(psrf[[1]][[2]])[2])) {
  points(psrf[[1]][[1]], psrf[[1]][[2]][,i], type="l", ylim=c(1,4), col=rgb(0,0,1,0.3))
}

for(i in 1:(dim(psrf[[2]][[2]])[2])) {
  points(psrf[[2]][[1]], psrf[[2]][[2]][,i], type="l", ylim=c(1,4), col=rgb(1,0,0,0.3))
}

write.table(cbind(psrf[[1]][[1]], psrf[[1]][[2]]), file=paste("multiclass_HMC_ard", OPTION_ARD, ".txt", sep=""), quote=FALSE, col.names=F, row.names=F)
write.table(cbind(psrf[[2]][[1]], psrf[[2]][[2]]), file=paste("multiclass_Gibbs_ard", OPTION_ARD, ".txt", sep=""), quote=FALSE, col.names=F, row.names=F)

legendHMC = paste("HMC - min ESS =", formatC(mean(ess[[1]]), format="e", digits=1), "- min TN-ESS =", formatC(mean(ess[[1]]/times[[1]]), format="e", digits=1))
legendGibbs = paste("Gibbs - min ESS =", formatC(mean(ess[[2]]), format="e", digits=1), "- min TN-ESS =", formatC(mean(ess[[2]]/times[[2]]), format="e", digits=1))

legend(min(psrf[[1]][[1]]), 4, c(legendHMC, legendGibbs), col=c(rgb(0,0,1,0.6),rgb(1,0,0,0.6)), lty=1)

dev.off()

}
