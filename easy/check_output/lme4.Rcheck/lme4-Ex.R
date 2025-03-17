pkgname <- "lme4"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
library('lme4')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("Arabidopsis")
### * Arabidopsis

flush(stderr()); flush(stdout())

### Name: Arabidopsis
### Title: Arabidopsis clipping/fertilization data
### Aliases: Arabidopsis
### Keywords: datasets

### ** Examples

data(Arabidopsis)
summary(Arabidopsis[,"total.fruits"])
table(gsub("[0-9].","",levels(Arabidopsis[,"popu"])))
library(lattice)
stripplot(log(total.fruits+1) ~ amd|nutrient, data = Arabidopsis,
          groups = gen,
          strip=strip.custom(strip.names=c(TRUE,TRUE)),
          type=c('p','a'), ## points and panel-average value --
          ## see ?panel.xyplot
          scales=list(x=list(rot=90)),
          main="Panel: nutrient, Color: genotype")



cleanEx()
nameEx("Dyestuff")
### * Dyestuff

flush(stderr()); flush(stdout())

### Name: Dyestuff
### Title: Yield of dyestuff by batch
### Aliases: Dyestuff Dyestuff2
### Keywords: datasets

### ** Examples

## Don't show: 
 # useful for the lme4-authors --- development, debugging, etc:
 commandArgs()[-1]
 if(FALSE) ## R environment variables:
 local({ ne <- names(e <- Sys.getenv())
         list(R    = e[grep("^R", ne)],
              "_R" = e[grep("^_R",ne)]) })
 Sys.getenv("R_ENVIRON")
 Sys.getenv("R_PROFILE")
 cat("R_LIBS:\n"); (RL <- strsplit(Sys.getenv("R_LIBS"), ":")[[1]])
 nRL <- normalizePath(RL)
 cat("and extra(:= not in R_LIBS) .libPaths():\n")
 .libPaths()[is.na(match(.libPaths(), nRL))]

 structure(Sys.info()[c(4,5,1:3)], class="simple.list") #-> 'nodename' ..
 sessionInfo()
 searchpaths()
 pkgI <- function(pkgname) {
   pd <- tryCatch(packageDescription(pkgname),
                  error=function(e)e, warning=function(w)w)
   if(inherits(pd, "error") || inherits(pd, "warning"))
     cat(sprintf("packageDescription(\"%s\") %s: %s\n",
                 pkgname, class(pd)[2], pd$message))
   else
     cat(sprintf("%s -- built: %s\n%*s -- dir  : %s\n",
                 pkgname, pd$Built, nchar(pkgname), "",
                 dirname(dirname(attr(pd, "file")))))
 }
 pkgI("Matrix")
 pkgI("Rcpp")
 ## 2012-03-12{MM}: fails with --as-cran
 pkgI("RcppEigen")
 pkgI("minqa")
 pkgI("lme4")
## End(Don't show)
require(lattice)
str(Dyestuff)
dotplot(reorder(Batch, Yield) ~ Yield, Dyestuff,
        ylab = "Batch", jitter.y = TRUE, aspect = 0.3,
        type = c("p", "a"))
dotplot(reorder(Batch, Yield) ~ Yield, Dyestuff2,
        ylab = "Batch", jitter.y = TRUE, aspect = 0.3,
        type = c("p", "a"))
(fm1 <- lmer(Yield ~ 1|Batch, Dyestuff))
(fm2 <- lmer(Yield ~ 1|Batch, Dyestuff2))



cleanEx()
nameEx("GHrule")
### * GHrule

flush(stderr()); flush(stdout())

### Name: GHrule
### Title: Univariate Gauss-Hermite quadrature rule
### Aliases: GHrule

### ** Examples

(r5  <- GHrule( 5, asMatrix=FALSE))
(r12 <- GHrule(12, asMatrix=FALSE))

## second, fourth, sixth, eighth and tenth central moments of the
## standard Gaussian N(0,1) density:
ps <- seq(2, 10, by = 2)
cbind(p = ps, "E[X^p]" = with(r5,  sapply(ps, function(p) sum(w * z^p)))) # p=10 is wrong for 5-rule
p <- 1:15
GQ12 <- with(r12, sapply(p, function(p) sum(w * z^p)))
cbind(p = p, "E[X^p]" = zapsmall(GQ12))
## standard R numerical integration can do it too:
intL <- lapply(p, function(p) integrate(function(x) x^p * dnorm(x),
                                        -Inf, Inf, rel.tol=1e-11))
integR <- sapply(intL, `[[`, "value")
cbind(p, "E[X^p]" = integR)# no zapsmall() needed here
all.equal(GQ12, integR, tol=0)# => shows small difference
stopifnot(all.equal(GQ12, integR, tol = 1e-10))
(xactMom <- cumprod(seq(1,13, by=2)))
stopifnot(all.equal(xactMom, GQ12[2*(1:7)], tol=1e-14))
## mean relative errors :
mean(abs(GQ12  [2*(1:7)] / xactMom - 1)) # 3.17e-16
mean(abs(integR[2*(1:7)] / xactMom - 1)) # 9.52e-17 {even better}



cleanEx()
nameEx("GQdk")
### * GQdk

flush(stderr()); flush(stdout())

### Name: GQdk
### Title: Sparse Gaussian / Gauss-Hermite Quadrature grid
### Aliases: GQdk GQN

### ** Examples

GQdk(2,5) # 53 x 3

GQN[[3]][[5]] # a 14 x 4 matrix



cleanEx()
nameEx("InstEval")
### * InstEval

flush(stderr()); flush(stdout())

### Name: InstEval
### Title: University Lecture/Instructor Evaluations by Students at ETH
### Aliases: InstEval
### Keywords: datasets

### ** Examples

str(InstEval)

head(InstEval, 16)
xtabs(~ service + dept, InstEval)



cleanEx()
nameEx("NelderMead-class")
### * NelderMead-class

flush(stderr()); flush(stdout())

### Name: NelderMead-class
### Title: Class '"NelderMead"' of Nelder-Mead optimizers and its Generator
### Aliases: NelderMead NelderMead-class
### Keywords: classes

### ** Examples

showClass("NelderMead")



cleanEx()
nameEx("Nelder_Mead")
### * Nelder_Mead

flush(stderr()); flush(stdout())

### Name: NelderMead
### Title: Nelder-Mead Optimization of Parameters, Possibly (Box)
###   Constrained
### Aliases: Nelder_Mead
### Keywords: classes

### ** Examples

fr <- function(x) {   ## Rosenbrock Banana function
    x1 <- x[1]
    x2 <- x[2]
    100 * (x2 - x1 * x1)^2 + (1 - x1)^2
}
p0 <- c(-1.2, 1)

oo  <- optim(p0, fr) ## also uses Nelder-Mead by default
o.  <- Nelder_Mead(fr, p0)
o.1 <- Nelder_Mead(fr, p0, control=list(verbose=1))# -> some iteration output
stopifnot(identical(o.[1:4], o.1[1:4]),
          all.equal(o.$par, oo$par, tolerance=1e-3))# diff: 0.0003865

o.2 <- Nelder_Mead(fr, p0, control=list(verbose=3, XtolRel=1e-15, FtolAbs= 1e-14))
all.equal(o.2[-5],o.1[-5], tolerance=1e-15)# TRUE, unexpectedly



cleanEx()
nameEx("Pastes")
### * Pastes

flush(stderr()); flush(stdout())

### Name: Pastes
### Title: Paste strength by batch and cask
### Aliases: Pastes
### Keywords: datasets

### ** Examples

str(Pastes)
require(lattice)
dotplot(cask ~ strength | reorder(batch, strength), Pastes,
        strip = FALSE, strip.left = TRUE, layout = c(1, 10),
        ylab = "Cask within batch",
        xlab = "Paste strength", jitter.y = TRUE)
## Modifying the factors to enhance the plot
Pastes <- within(Pastes, batch <- reorder(batch, strength))
Pastes <- within(Pastes, sample <- reorder(reorder(sample, strength),
          as.numeric(batch)))
dotplot(sample ~ strength | batch, Pastes,
        strip = FALSE, strip.left = TRUE, layout = c(1, 10),
        scales = list(y = list(relation = "free")),
        ylab = "Sample within batch",
        xlab = "Paste strength", jitter.y = TRUE)
## Four equivalent models differing only in specification
(fm1 <- lmer(strength ~ (1|batch) + (1|sample), Pastes))
(fm2 <- lmer(strength ~ (1|batch/cask), Pastes))
(fm3 <- lmer(strength ~ (1|batch) + (1|batch:cask), Pastes))
(fm4 <- lmer(strength ~ (1|batch/sample), Pastes))
## fm4 results in redundant labels on the sample:batch interaction
head(ranef(fm4)[[1]])
## compare to fm1
head(ranef(fm1)[[1]])
## This model is different and NOT appropriate for these data
(fm5 <- lmer(strength ~ (1|batch) + (1|cask), Pastes))

L <- getME(fm1, "L")
Matrix::image(L, sub = "Structure of random effects interaction in pastes model")



cleanEx()
nameEx("Penicillin")
### * Penicillin

flush(stderr()); flush(stdout())

### Name: Penicillin
### Title: Variation in penicillin testing
### Aliases: Penicillin
### Keywords: datasets

### ** Examples

str(Penicillin)
require(lattice)
dotplot(reorder(plate, diameter) ~ diameter, Penicillin, groups = sample,
        ylab = "Plate", xlab = "Diameter of growth inhibition zone (mm)",
        type = c("p", "a"), auto.key = list(columns = 3, lines = TRUE,
        title = "Penicillin sample"))
(fm1 <- lmer(diameter ~ (1|plate) + (1|sample), Penicillin))

L <- getME(fm1, "L")
Matrix::image(L, main = "L",
              sub = "Penicillin: Structure of random effects interaction")



cleanEx()
nameEx("VarCorr")
### * VarCorr

flush(stderr()); flush(stdout())

### Name: VarCorr
### Title: Extract Variance and Correlation Components
### Aliases: VarCorr VarCorr.merMod as.data.frame.VarCorr.merMod
###   print.VarCorr.merMod
### Keywords: models

### ** Examples

data(Orthodont, package="nlme")
fm1 <- lmer(distance ~ age + (age|Subject), data = Orthodont)
print(vc <- VarCorr(fm1))  ## default print method: standard dev and corr
## both variance and std.dev.
print(vc,comp=c("Variance","Std.Dev."), digits=2)
## variance only
print(vc, comp=c("Variance"))
## standard deviations only, but covariances rather than correlations
print(vc, corr = FALSE)
as.data.frame(vc)
as.data.frame(vc, order="lower.tri")



cleanEx()
nameEx("VerbAgg")
### * VerbAgg

flush(stderr()); flush(stdout())

### Name: VerbAgg
### Title: Verbal Aggression item responses
### Aliases: VerbAgg
### Keywords: datasets

### ** Examples

str(VerbAgg)
## Show how  r2 := h(resp) is defined:
with(VerbAgg, stopifnot( identical(r2, {
     r <- factor(resp, ordered=FALSE); levels(r) <- c("N","Y","Y"); r})))

xtabs(~ item + resp, VerbAgg)
xtabs(~ btype + resp, VerbAgg)
round(100 * ftable(prop.table(xtabs(~ situ + mode + resp, VerbAgg), 1:2), 1))
person <- unique(subset(VerbAgg, select = c(id, Gender, Anger)))
require(lattice)
densityplot(~ Anger, person, groups = Gender, auto.key = list(columns = 2),
            xlab = "Trait Anger score (STAXI)")

if(lme4:::testLevel() >= 3) { ## takes about 15 sec
    print(fmVA <- glmer(r2 ~ (Anger + Gender + btype + situ)^2 +
 		   (1|id) + (1|item), family = binomial, data =
		   VerbAgg), corr=FALSE)
} ## testLevel() >= 3
if (interactive()) {
## much faster but less accurate
    print(fmVA0 <- glmer(r2 ~ (Anger + Gender + btype + situ)^2 +
                             (1|id) + (1|item), family = binomial,
                         data = VerbAgg, nAGQ=0L), corr=FALSE)
} ## interactive()



cleanEx()
nameEx("allFit")
### * allFit

flush(stderr()); flush(stdout())

### Name: allFit
### Title: Refit a fitted model with all available optimizers
### Aliases: allFit
### Keywords: models

### ** Examples

if (interactive()) {
library(lme4)
  gm1 <- glmer(cbind(incidence, size - incidence) ~ period + (1 | herd),
             data = cbpp, family = binomial)
  ## show available methods
  allFit(show.meth.tab=TRUE) 
  gm_all <- allFit(gm1)
  ss <- summary(gm_all)
  ss$which.OK            ## logical vector: which optimizers worked?
  ## the other components only contain values for the optimizers that worked
  ss$llik                ## vector of log-likelihoods
  ss$fixef               ## table of fixed effects
  ss$sdcor               ## table of random effect SDs and correlations
  ss$theta               ## table of random effects parameters, Cholesky scale
} 
## Not run: 
##D   ## Parallel examples for Windows
##D   nc <- detectCores()-1
##D   optCls <- makeCluster(nc, type = "SOCK")
##D   clusterEvalQ(optCls,library("lme4"))
##D   ### not necessary here because using a built-in
##D   ## data set, but in general you should clusterExport() your data
##D   clusterExport(optCls, "cbpp")
##D   system.time(af1 <- allFit(m0, parallel = 'snow', 
##D                           ncpus = nc, cl=optCls))
##D   stopCluster(optCls)
## End(Not run)
 



cleanEx()
nameEx("bootMer")
### * bootMer

flush(stderr()); flush(stdout())

### Name: bootMer
### Title: Model-based (Semi-)Parametric Bootstrap for Mixed Models
### Aliases: bootMer
### Keywords: htest models

### ** Examples

if (interactive()) {
fm01ML <- lmer(Yield ~ 1|Batch, Dyestuff, REML = FALSE)
## see ?"profile-methods"
mySumm <- function(.) { s <- sigma(.)
    c(beta =getME(., "beta"), sigma = s, sig01 = unname(s * getME(., "theta"))) }
(t0 <- mySumm(fm01ML)) # just three parameters
## alternatively:
mySumm2 <- function(.) {
    c(beta=fixef(.),sigma=sigma(.), sig01=sqrt(unlist(VarCorr(.))))
}

set.seed(101)
## 3.8s (on a 5600 MIPS 64bit fast(year 2009) desktop "AMD Phenom(tm) II X4 925"):
system.time( boo01 <- bootMer(fm01ML, mySumm, nsim = 100) )

## to "look" at it
if (requireNamespace("boot")) {
    boo01
    ## note large estimated bias for sig01
    ## (~30% low, decreases _slightly_ for nsim = 1000)

    ## extract the bootstrapped values as a data frame ...
    head(as.data.frame(boo01))

    ## ------ Bootstrap-based confidence intervals ------------

    ## warnings about "Some ... intervals may be unstable" go away
    ##   for larger bootstrap samples, e.g. nsim=500

    ## intercept
    (bCI.1 <- boot::boot.ci(boo01, index=1, type=c("norm", "basic", "perc")))# beta

    ## Residual standard deviation - original scale:
    (bCI.2  <- boot::boot.ci(boo01, index=2, type=c("norm", "basic", "perc")))
    ## Residual SD - transform to log scale:
    (bCI.2L <- boot::boot.ci(boo01, index=2, type=c("norm", "basic", "perc"),
                       h = log, hdot = function(.) 1/., hinv = exp))

    ## Among-batch variance:
    (bCI.3 <- boot::boot.ci(boo01, index=3, type=c("norm", "basic", "perc"))) # sig01

    
    confint(boo01)
    confint(boo01,type="norm")
    confint(boo01,type="basic")

    ## Graphical examination:
    plot(boo01,index=3)

    ## Check stored values from a longer (1000-replicate) run:
    (load(system.file("testdata","boo01L.RData", package="lme4")))# "boo01L"
    plot(boo01L, index=3)
    mean(boo01L$t[,"sig01"]==0) ## note point mass at zero!
} 
} 



cleanEx()
nameEx("cake")
### * cake

flush(stderr()); flush(stdout())

### Name: cake
### Title: Breakage Angle of Chocolate Cakes
### Aliases: cake
### Keywords: datasets

### ** Examples

str(cake)
## 'temp' is continuous, 'temperature' an ordered factor with 6 levels

(fm1 <- lmer(angle ~ recipe * temperature + (1|recipe:replicate), cake, REML= FALSE))
(fm2 <- lmer(angle ~ recipe + temperature + (1|recipe:replicate), cake, REML= FALSE))
(fm3 <- lmer(angle ~ recipe + temp        + (1|recipe:replicate), cake, REML= FALSE))

## and now "choose" :
anova(fm3, fm2, fm1)



cleanEx()
nameEx("cbpp")
### * cbpp

flush(stderr()); flush(stdout())

### Name: cbpp
### Title: Contagious bovine pleuropneumonia
### Aliases: cbpp
### Keywords: datasets

### ** Examples

## response as a matrix
(m1 <- glmer(cbind(incidence, size - incidence) ~ period + (1 | herd),
             family = binomial, data = cbpp))
## response as a vector of probabilities and usage of argument "weights"
m1p <- glmer(incidence / size ~ period + (1 | herd), weights = size,
             family = binomial, data = cbpp)
## Confirm that these are equivalent:
stopifnot(all.equal(fixef(m1), fixef(m1p), tolerance = 1e-5),
          all.equal(ranef(m1), ranef(m1p), tolerance = 1e-5))

## GLMM with individual-level variability (accounting for overdispersion)
cbpp$obs <- 1:nrow(cbpp)
(m2 <- glmer(cbind(incidence, size - incidence) ~ period + (1 | herd) +  (1|obs),
              family = binomial, data = cbpp))



cleanEx()
nameEx("confint.merMod")
### * confint.merMod

flush(stderr()); flush(stdout())

### Name: confint.merMod
### Title: Compute Confidence Intervals for Parameters of a [ng]lmer Fit
### Aliases: confint.merMod confint.thpr

### ** Examples

if (interactive() || lme4_testlevel() >= 3) {
fm1 <- lmer(Reaction ~ Days + (Days|Subject), sleepstudy)
fm1W <- confint(fm1, method="Wald")# very fast, but not useful for "sigmas" = var-cov pars
fm1W
(fm2 <- lmer(Reaction ~ Days + (Days || Subject), sleepstudy))
(CI2 <- confint(fm2, maxpts = 8)) # method = "profile"; 8: to be much faster
## Don't show: 
 stopifnot(all.equal(tolerance = 5e-6, signif(unname(CI2), 7),
               array(c(15.25847, 3.964157, 22.88062, 237.5732,  7.33431,
                       37.78184, 8.768238, 28.78768, 265.2383, 13.60057),
                     dim = c(5L, 2L))))
## End(Don't show)
if (lme4_testlevel() >= 3) {
  system.time(fm1P <- confint(fm1, method="profile", ## <- default
                              oldNames = FALSE))
  ## --> ~ 2.2 seconds (2022)
  set.seed(123) # (reproducibility when using bootstrap)
  system.time(fm1B <- confint(fm1, method="boot", oldNames=FALSE,
                              .progress="txt", PBargs= list(style=3)))
  ## --> ~ 6.2 seconds (2022) and warning, messages
} else {
    load(system.file("testdata","confint_ex.rda",package="lme4"))
}
fm1P
fm1B
} ## if interactive && testlevel>=3



cleanEx()
nameEx("convergence")
### * convergence

flush(stderr()); flush(stdout())

### Name: convergence
### Title: Assessing Convergence for Fitted Models
### Aliases: convergence

### ** Examples

if (interactive()) {
fm1 <- lmer(Reaction ~ Days + (Days | Subject), sleepstudy)

## 1. decrease stopping tolerances
strict_tol <- lmerControl(optCtrl=list(xtol_abs=1e-8, ftol_abs=1e-8))
if (all(fm1@optinfo$optimizer=="nloptwrap")) {
    fm1.tol <- update(fm1, control=strict_tol)
}

## 2. center and scale predictors:
ss.CS <- transform(sleepstudy, Days=scale(Days))
fm1.CS <- update(fm1, data=ss.CS)

## 3. recompute gradient and Hessian with Richardson extrapolation
devfun <- update(fm1, devFunOnly=TRUE)
if (isLMM(fm1)) {
    pars <- getME(fm1,"theta")
} else {
    ## GLMM: requires both random and fixed parameters
    pars <- getME(fm1, c("theta","fixef"))
}
if (require("numDeriv")) {
    cat("hess:\n"); print(hess <- hessian(devfun, unlist(pars)))
    cat("grad:\n"); print(grad <- grad(devfun, unlist(pars)))
    cat("scaled gradient:\n")
    print(scgrad <- solve(chol(hess), grad))
}
## compare with internal calculations:
fm1@optinfo$derivs

## compute reciprocal condition number of Hessian
H <- fm1@optinfo$derivs$Hessian
Matrix::rcond(H)

## 4. restart the fit from the original value (or
## a slightly perturbed value):
fm1.restart <- update(fm1, start=pars)
set.seed(101)
pars_x <- runif(length(pars),pars/1.01,pars*1.01)
fm1.restart2 <- update(fm1, start=pars_x,
                       control=strict_tol)

## 5. try all available optimizers

  fm1.all <- allFit(fm1)
  ss <- summary(fm1.all)
  ss$ fixef               ## fixed effects
  ss$ llik                ## log-likelihoods
  ss$ sdcor               ## SDs and correlations
  ss$ theta               ## Cholesky factors
  ss$ which.OK            ## which fits worked

} 



cleanEx()
nameEx("devfun2")
### * devfun2

flush(stderr()); flush(stdout())

### Name: devfun2
### Title: Deviance Function in Terms of Standard Deviations/Correlations
### Aliases: devfun2
### Keywords: utilities

### ** Examples

m1 <- lmer(Reaction~Days+(Days|Subject),sleepstudy)
dd <- devfun2(m1, useSc=TRUE)
pp <- attr(dd,"optimum")
## extract variance-covariance and residual std dev parameters
sigpars <- pp[grepl("^\\.sig",names(pp))]
all.equal(unname(dd(sigpars)),deviance(refitML(m1)))



cleanEx()
nameEx("drop1.merMod")
### * drop1.merMod

flush(stderr()); flush(stdout())

### Name: drop1.merMod
### Title: Drop all possible single fixed-effect terms from a mixed effect
###   model
### Aliases: drop1.merMod
### Keywords: misc

### ** Examples

fm1 <- lmer(Reaction ~ Days + (Days|Subject), sleepstudy)
## likelihood ratio tests
drop1(fm1,test="Chisq")
## use Kenward-Roger corrected F test, or parametric bootstrap,
## to test the significance of each dropped predictor
if (require(pbkrtest) && packageVersion("pbkrtest")>="0.3.8") {
   KRSumFun <- function(object, objectDrop, ...) {
      krnames <- c("ndf","ddf","Fstat","p.value","F.scaling")
      r <- if (missing(objectDrop)) {
          setNames(rep(NA,length(krnames)),krnames)
      } else {
         krtest <- KRmodcomp(object,objectDrop)
         unlist(krtest$stats[krnames])
      }
      attr(r,"method") <- c("Kenward-Roger via pbkrtest package")
      r
   }
   drop1(fm1, test="user", sumFun=KRSumFun)

   if(lme4:::testLevel() >= 3) { ## takes about 16 sec
     nsim <- 100
     PBSumFun <- function(object, objectDrop, ...) {
	pbnames <- c("stat","p.value")
	r <- if (missing(objectDrop)) {
	    setNames(rep(NA,length(pbnames)),pbnames)
	} else {
	   pbtest <- PBmodcomp(object,objectDrop,nsim=nsim)
	   unlist(pbtest$test[2,pbnames])
	}
	attr(r,"method") <- c("Parametric bootstrap via pbkrtest package")
	r
     }
     system.time(drop1(fm1, test="user", sumFun=PBSumFun))
   }
}
## workaround for creating a formula in a separate environment
createFormula <- function(resp, fixed, rand) {  
    f <- reformulate(c(fixed,rand),response=resp)
    ## use the parent (createModel) environment, not the
    ## environment of this function (which does not contain 'data')
    environment(f) <- parent.frame()
    f
}
createModel <- function(data) {
    mf.final <- createFormula("Reaction", "Days", "(Days|Subject)")
    lmer(mf.final, data=data)
}
drop1(createModel(data=sleepstudy))



cleanEx()
nameEx("dummy")
### * dummy

flush(stderr()); flush(stdout())

### Name: dummy
### Title: Dummy variables (experimental)
### Aliases: dummy

### ** Examples

data(Orthodont,package="nlme")
lmer(distance ~ age + (age|Subject) +
     (0+dummy(Sex, "Female")|Subject), data = Orthodont)



cleanEx()
nameEx("expandDoubleVerts")
### * expandDoubleVerts

flush(stderr()); flush(stdout())

### Name: expandDoubleVerts
### Title: Expand terms with "||" notation into separate "|" terms
### Aliases: expandDoubleVerts ||
### Keywords: models utilities

### ** Examples

  m <- ~ x + (x || g)
  expandDoubleVerts(m)
  set.seed(101)
  dd <- expand.grid(f=factor(letters[1:3]),g=factor(1:200),rep=1:3)
  dd$y <- simulate(~f + (1|g) + (0+dummy(f,"b")|g) + (0+dummy(f,"c")|g),
          newdata=dd,
          newparams=list(beta=rep(0,3),
                         theta=c(1,2,1),
                         sigma=1),
          family=gaussian)[[1]]
  m1 <- lmer(y~f+(f|g),data=dd)
  VarCorr(m1)
  m2 <- lmer(y~f+(1|g) + (0+dummy(f,"b")|g) + (0+dummy(f,"c")|g),
               data=dd)
  VarCorr(m2)



cleanEx()
nameEx("findbars")
### * findbars

flush(stderr()); flush(stdout())

### Name: findbars
### Title: Determine random-effects expressions from a formula
### Aliases: findbars
### Keywords: models utilities

### ** Examples

findbars(f1 <- Reaction ~ Days + (Days | Subject))
## => list( Days | Subject )
## These two are equivalent:% tests in ../inst/tests/test-doubleVertNotation.R
findbars(y ~ Days + (1 | Subject) + (0 + Days | Subject))
findbars(y ~ Days + (Days || Subject))
## => list of length 2:  list ( 1 | Subject ,  0 + Days | Subject)
findbars(~ 1 + (1 | batch / cask))
## => list of length 2:  list ( 1 | cask:batch ,  1 | batch)
## Don't show: 
stopifnot(identical(findbars(f1),
                    list(quote(Days | Subject))))
## End(Don't show)



cleanEx()
nameEx("fixef")
### * fixef

flush(stderr()); flush(stdout())

### Name: fixef
### Title: Extract fixed-effects estimates
### Aliases: fixed.effects fixef fixef.merMod
### Keywords: models

### ** Examples

fixef(lmer(Reaction ~ Days + (1|Subject) + (0+Days|Subject), sleepstudy))
fm2 <- lmer(Reaction ~ Days + Days2 + (1|Subject),
            data=transform(sleepstudy,Days2=Days))
fixef(fm2,add.dropped=TRUE)
## first two parameters are the same ...
stopifnot(all.equal(fixef(fm2,add.dropped=TRUE)[1:2],
                    fixef(fm2)))



cleanEx()
nameEx("fortify")
### * fortify

flush(stderr()); flush(stdout())

### Name: fortify
### Title: add information to data based on a fitted model
### Aliases: fortify fortify.merMod getData getData.merMod

### ** Examples

  fm1 <- lmer(Reaction~Days+(1|Subject),sleepstudy)
  names(fortify.merMod(fm1))



cleanEx()
nameEx("getME")
### * getME

flush(stderr()); flush(stdout())

### Name: getME
### Title: Extract or Get Generalized Components from a Fitted Mixed
###   Effects Model
### Aliases: getL getL,merMod-method getME getME.merMod
### Keywords: utilities

### ** Examples

## shows many methods you should consider *before* using getME():
methods(class = "merMod")

(fm1 <- lmer(Reaction ~ Days + (Days|Subject), sleepstudy))
Z <- getME(fm1, "Z")
stopifnot(is(Z, "CsparseMatrix"),
          c(180,36) == dim(Z),
	  all.equal(fixef(fm1), b1 <- getME(fm1, "beta"),
		    check.attributes=FALSE, tolerance = 0))

## A way to get *all* getME()s :
## internal consistency check ensuring that all work:
parts <- getME(fm1, "ALL")
str(parts, max=2)
stopifnot(identical(Z,  parts $ Z),
          identical(b1, parts $ beta))



cleanEx()
nameEx("glmFamily-class")
### * glmFamily-class

flush(stderr()); flush(stdout())

### Name: glmFamily-class
### Title: Class '"glmFamily"' - a reference class for 'family'
### Aliases: glmFamily-class
### Keywords: classes

### ** Examples

str(glmFamily$new(family=poisson()))



cleanEx()
nameEx("glmer")
### * glmer

flush(stderr()); flush(stdout())

### Name: glmer
### Title: Fitting Generalized Linear Mixed-Effects Models
### Aliases: glmer
### Keywords: models

### ** Examples

## generalized linear mixed model
library(lattice)
xyplot(incidence/size ~ period|herd, cbpp, type=c('g','p','l'),
       layout=c(3,5), index.cond = function(x,y)max(y))
(gm1 <- glmer(cbind(incidence, size - incidence) ~ period + (1 | herd),
              data = cbpp, family = binomial))
## using nAGQ=0 only gets close to the optimum
(gm1a <- glmer(cbind(incidence, size - incidence) ~ period + (1 | herd),
               cbpp, binomial, nAGQ = 0))
## using  nAGQ = 9  provides a better evaluation of the deviance
## Currently the internal calculations use the sum of deviance residuals,
## which is not directly comparable with the nAGQ=0 or nAGQ=1 result.
## 'verbose = 1' monitors iteratin a bit; (verbose = 2 does more):
(gm1a <- glmer(cbind(incidence, size - incidence) ~ period + (1 | herd),
               cbpp, binomial, verbose = 1, nAGQ = 9))

## GLMM with individual-level variability (accounting for overdispersion)
## For this data set the model is the same as one allowing for a period:herd
## interaction, which the plot indicates could be needed.
cbpp$obs <- 1:nrow(cbpp)
(gm2 <- glmer(cbind(incidence, size - incidence) ~ period +
    (1 | herd) +  (1|obs),
              family = binomial, data = cbpp))
anova(gm1,gm2)

## glmer and glm log-likelihoods are consistent
gm1Devfun <- update(gm1,devFunOnly=TRUE)
gm0 <- glm(cbind(incidence, size - incidence) ~ period,
           family = binomial, data = cbpp)
## evaluate GLMM deviance at RE variance=theta=0, beta=(GLM coeffs)
gm1Dev0 <- gm1Devfun(c(0,coef(gm0)))
## compare
stopifnot(all.equal(gm1Dev0,c(-2*logLik(gm0))))
## the toenail oncholysis data from Backer et al 1998
## these data are notoriously difficult to fit
## Not run: 
##D if (require("HSAUR3")) {
##D     gm2 <- glmer(outcome~treatment*visit+(1|patientID),
##D                  data=toenail,
##D                  family=binomial,nAGQ=20)
##D }
## End(Not run)



cleanEx()
nameEx("glmer.nb")
### * glmer.nb

flush(stderr()); flush(stdout())

### Name: glmer.nb
### Title: Fitting Negative Binomial GLMMs
### Aliases: glmer.nb negative.binomial
### Keywords: models

### ** Examples

set.seed(101)
dd <- expand.grid(f1 = factor(1:3),
                  f2 = LETTERS[1:2], g=1:9, rep=1:15,
          KEEP.OUT.ATTRS=FALSE)
summary(mu <- 5*(-4 + with(dd, as.integer(f1) + 4*as.numeric(f2))))
dd$y <- rnbinom(nrow(dd), mu = mu, size = 0.5)
str(dd)
require("MASS")## and use its glm.nb() - as indeed we have zero random effect:
## Not run: 
##D m.glm <- glm.nb(y ~ f1*f2, data=dd, trace=TRUE)
##D summary(m.glm)
##D m.nb <- glmer.nb(y ~ f1*f2 + (1|g), data=dd, verbose=TRUE)
##D m.nb
##D ## The neg.binomial theta parameter:
##D getME(m.nb, "glmer.nb.theta")
##D LL <- logLik(m.nb)
##D ## mixed model has 1 additional parameter (RE variance)
##D stopifnot(attr(LL,"df")==attr(logLik(m.glm),"df")+1)
##D plot(m.nb, resid(.) ~ g)# works, as long as data 'dd' is found
## End(Not run)



cleanEx()
nameEx("golden-class")
### * golden-class

flush(stderr()); flush(stdout())

### Name: golden-class
### Title: Class '"golden"' and Generator for Golden Search Optimizer Class
### Aliases: golden-class golden
### Keywords: classes

### ** Examples

showClass("golden")

golden(lower= -100, upper= 1e100)



cleanEx()
nameEx("grouseticks")
### * grouseticks

flush(stderr()); flush(stdout())

### Name: grouseticks
### Title: Data on red grouse ticks from Elston et al. 2001
### Aliases: grouseticks grouseticks_agg
### Keywords: datasets

### ** Examples

if (interactive()) {
data(grouseticks)
## Figure 1a from Elston et al
par(las=1,bty="l")
tvec <- c(0,1,2,5,20,40,80)
pvec <- c(4,1,3)
with(grouseticks_agg,plot(1+meanTICKS~HEIGHT,
                  pch=pvec[factor(YEAR)],
                  log="y",axes=FALSE,
                  xlab="Altitude (m)",
                  ylab="Brood mean ticks"))
axis(side=1)
axis(side=2,at=tvec+1,label=tvec)
box()
abline(v=405,lty=2)
## Figure 1b
with(grouseticks_agg,plot(varTICKS~meanTICKS,
                  pch=4,
                  xlab="Brood mean ticks",
                  ylab="Within-brood variance"))
curve(1*x,from=0,to=70,add=TRUE)
## Model fitting
form <- TICKS~YEAR+HEIGHT+(1|BROOD)+(1|INDEX)+(1|LOCATION)
(full_mod1  <- glmer(form, family="poisson",data=grouseticks))
}



graphics::par(get("par.postscript", pos = 'CheckExEnv'))
cleanEx()
nameEx("hatvalues.merMod")
### * hatvalues.merMod

flush(stderr()); flush(stdout())

### Name: hatvalues.merMod
### Title: Diagonal elements of the hat matrix
### Aliases: hatvalues.merMod

### ** Examples

m <- lmer(Reaction ~ Days + (Days | Subject), sleepstudy)
hatvalues(m)



cleanEx()
nameEx("influence.merMod")
### * influence.merMod

flush(stderr()); flush(stdout())

### Name: influence.merMod
### Title: Influence Diagnostics for Mixed-Effects Models
### Aliases: influence.merMod dfbeta.influence.merMod
###   dfbetas.influence.merMod cooks.distance.influence.merMod
###   cooks.distance.merMod
### Keywords: models

### ** Examples

if (interactive()) {
  fm1 <- lmer(Reaction ~ Days + (Days | Subject), sleepstudy)
  inf_fm1 <- influence(fm1, "Subject")
  if (require("car")) {
    infIndexPlot(inf_fm1)
  }
  dfbeta(inf_fm1)
  dfbetas(inf_fm1)
  gm1 <- glmer(cbind(incidence, size - incidence) ~ period + (1 | herd),
               data = cbpp, family = binomial)
  inf_gm1 <- influence(gm1, "herd", maxfun=100)
  gm1.11 <- update(gm1, subset = herd != 11) # check deleting herd 11
  if (require("car")) {
    infIndexPlot(inf_gm1)
    compareCoefs(gm1, gm1.11)
  }
  if(packageVersion("car") >= "3.0.10") {
    dfbeta(inf_gm1)
    dfbetas(inf_gm1)
  }
 } 



cleanEx()
nameEx("isNested")
### * isNested

flush(stderr()); flush(stdout())

### Name: isNested
### Title: Is f1 nested within f2?
### Aliases: isNested

### ** Examples

with(Pastes, isNested(cask, batch))   ## => FALSE
with(Pastes, isNested(sample, batch))  ## => TRUE



cleanEx()
nameEx("isREML")
### * isREML

flush(stderr()); flush(stdout())

### Name: isREML
### Title: Check characteristics of models
### Aliases: isGLMM isLMM isNLMM isREML isGLMM.merMod isLMM.merMod
###   isNLMM.merMod isREML.merMod

### ** Examples

fm1 <- lmer(Reaction ~ Days + (Days|Subject), sleepstudy)
gm1 <- glmer(cbind(incidence, size - incidence) ~ period + (1 | herd),
              data = cbpp, family = binomial)
nm1 <- nlmer(circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym|Tree,
             Orange, start = c(Asym = 200, xmid = 725, scal = 350))

isLMM(fm1)
isGLMM(gm1)
## check all :
is.MM <- function(x) c(LMM = isLMM(x), GLMM= isGLMM(x), NLMM= isNLMM(x))
stopifnot(cbind(is.MM(fm1), is.MM(gm1), is.MM(nm1))
	  == diag(rep(TRUE,3)))



cleanEx()
nameEx("lmList")
### * lmList

flush(stderr()); flush(stdout())

### Name: lmList
### Title: Fit List of lm or glm Objects with a Common Model
### Aliases: lmList plot.lmList
### Keywords: models

### ** Examples

fm.plm  <- lmList(Reaction ~ Days | Subject, sleepstudy)
coef(fm.plm)
fm.2  <- update(fm.plm, pool = FALSE)
## coefficients are the same, "pooled or unpooled":
stopifnot( all.equal(coef(fm.2), coef(fm.plm)) )

(ci <- confint(fm.plm)) # print and rather *see* :
plot(ci)                # how widely they vary for the individuals



cleanEx()
nameEx("lmList4-class")
### * lmList4-class

flush(stderr()); flush(stdout())

### Name: lmList4-class
### Title: Class "lmList4" of 'lm' Objects on Common Model
### Aliases: lmList4-class show,lmList4-method
### Keywords: classes

### ** Examples

if(getRversion() >= "3.2.0") {
  (mm <- methods(class = "lmList4"))
  ## The S3 ("not S4") ones :
  mm[!attr(mm,"info")[,"isS4"]]
}
## For more examples:  example(lmList)  i.e., ?lmList



cleanEx()
nameEx("lmResp-class")
### * lmResp-class

flush(stderr()); flush(stdout())

### Name: lmResp-class
### Title: Reference Classes for Response Modules,
###   '"(lm|glm|nls|lmer)Resp"'
### Aliases: glmResp-class lmerResp-class lmResp-class nlsResp-class
### Keywords: classes

### ** Examples

showClass("lmResp")
str(lmResp$new(y=1:4))
showClass("glmResp")
str(glmResp$new(family=poisson(), y=1:4))
showClass("nlsResp")
showClass("lmerResp")
str(lmerResp$new(y=1:4))



cleanEx()
nameEx("lmer")
### * lmer

flush(stderr()); flush(stdout())

### Name: lmer
### Title: Fit Linear Mixed-Effects Models
### Aliases: lmer
### Keywords: models

### ** Examples

## linear mixed models - reference values from older code
(fm1 <- lmer(Reaction ~ Days + (Days | Subject), sleepstudy))
summary(fm1)# (with its own print method; see class?merMod % ./merMod-class.Rd

str(terms(fm1))
stopifnot(identical(terms(fm1, fixed.only=FALSE),
                    terms(model.frame(fm1))))
attr(terms(fm1, FALSE), "dataClasses") # fixed.only=FALSE needed for dataCl.

## Maximum Likelihood (ML), and "monitor" iterations via 'verbose':
fm1_ML <- update(fm1, REML=FALSE, verbose = 1)
(fm2 <- lmer(Reaction ~ Days + (Days || Subject), sleepstudy))
anova(fm1, fm2)
sm2 <- summary(fm2)
print(fm2, digits=7, ranef.comp="Var") # the print.merMod()         method
print(sm2, digits=3, corr=FALSE)       # the print.summary.merMod() method

## Fit sex-specific variances by constructing numeric dummy variables
## for sex and sex:age; in this case the estimated variance differences
## between groups in both intercept and slope are zero ...
data(Orthodont,package="nlme")
Orthodont$nsex <- as.numeric(Orthodont$Sex=="Male")
Orthodont$nsexage <- with(Orthodont, nsex*age)
lmer(distance ~ age + (age|Subject) + (0+nsex|Subject) +
     (0 + nsexage|Subject), data=Orthodont)



cleanEx()
nameEx("lmerControl")
### * lmerControl

flush(stderr()); flush(stdout())

### Name: lmerControl
### Title: Control of Mixed Model Fitting
### Aliases: glmerControl lmerControl nlmerControl .makeCC

### ** Examples

str(lmerControl())
str(glmerControl())
## fit with default algorithm [nloptr version of BOBYQA] ...
fm0 <- lmer(Reaction ~ Days +   ( 1 | Subject), sleepstudy)
fm1 <- lmer(Reaction ~ Days + (Days | Subject), sleepstudy)
## or with "bobyqa" (default 2013 - 2019-02) ...
fm1_bobyqa <- update(fm1, control = lmerControl(optimizer="bobyqa"))
## or with "Nelder_Mead" (the default till 2013) ...
fm1_NMead <- update(fm1, control = lmerControl(optimizer="Nelder_Mead"))
## or with the nlminb function used in older (<1.0) versions of lme4;
## this will usually replicate older results
if (require(optimx)) {
    fm1_nlminb <- update(fm1,
                         control = lmerControl(optimizer= "optimx",
                                               optCtrl  = list(method="nlminb")))
    ## The other option here is method="L-BFGS-B".
}

## Or we can wrap base::optim():
optimwrap <- function(fn,par,lower,upper,control=list(),
                      ...) {
    if (is.null(control$method)) stop("must specify method in optCtrl")
    method <- control$method
    control$method <- NULL
    ## "Brent" requires finite upper values (lower bound will always
    ##  be zero in this case)
    if (method=="Brent") upper <- pmin(1e4,upper)
    res <- optim(par=par, fn=fn, lower=lower,upper=upper,
                 control=control,method=method,...)
    with(res, list(par  = par,
                   fval = value,
                   feval= counts[1],
                   conv = convergence,
                   message = message))
}
fm0_brent <- update(fm0,
                    control = lmerControl(optimizer = "optimwrap",
                                          optCtrl = list(method="Brent")))

## You can also use functions (in addition to the lmerControl() default "NLOPT_BOBYQA")
## from the 'nloptr' package, see also  '?nloptwrap' :
if (require(nloptr)) {
    fm1_nloptr_NM <- update(fm1, control=lmerControl(optimizer="nloptwrap",
                                      optCtrl=list(algorithm="NLOPT_LN_NELDERMEAD")))
    fm1_nloptr_COBYLA <- update(fm1, control=lmerControl(optimizer="nloptwrap",
                                      optCtrl=list(algorithm="NLOPT_LN_COBYLA",
                                                   xtol_rel=1e-6,
                                                   xtol_abs=1e-10,
                                                   ftol_abs=1e-10)))
}
## other algorithm options include NLOPT_LN_SBPLX



cleanEx()
nameEx("merMod-class")
### * merMod-class

flush(stderr()); flush(stdout())

### Name: merMod-class
### Title: Class "merMod" of Fitted Mixed-Effect Models
### Aliases: anova.merMod as.function.merMod coef.merMod deviance.merMod
###   df.residual.merMod extractAIC.merMod family.merMod fitted.merMod
###   formula.merMod glmerMod-class lmerMod-class logLik.merMod merMod
###   merMod-class model.frame.merMod model.matrix.merMod ngrps.merMod
###   nobs.merMod nobs nlmerMod-class print.merMod print.summary.merMod
###   show,merMod-method show.merMod show.summary.merMod summary.merMod
###   summary.summary.merMod terms.merMod update.merMod weights.merMod
###   REMLcrit
### Keywords: classes

### ** Examples

showClass("merMod")
methods(class="merMod")## over 30  (S3) methods available

m1 <- lmer(Reaction ~ Days + (Days | Subject), sleepstudy)
print(m1, ranef.corr = TRUE)   ## print correlations of REs
print(m1, ranef.corr = FALSE)  ## print covariances of REs




cleanEx()
nameEx("merPredD-class")
### * merPredD-class

flush(stderr()); flush(stdout())

### Name: merPredD-class
### Title: Class '"merPredD"' - a Dense Predictor Reference Class
### Aliases: merPredD-class
### Keywords: classes

### ** Examples

showClass("merPredD")
pp <- slot(lmer(Yield ~ 1|Batch, Dyestuff), "pp")
stopifnot(is(pp, "merPredD"))
str(pp) # an overview of all fields and methods' names.



cleanEx()
nameEx("mkReTrms")
### * mkReTrms

flush(stderr()); flush(stdout())

### Name: mkReTrms
### Title: Make Random Effect Terms: Create Z, Lambda, Lind, etc.
### Aliases: mkReTrms mkNewReTrms
### Keywords: utilities

### ** Examples

data("Pixel", package="nlme")
mform <- pixel ~ day + I(day^2) + (day | Dog) + (1 | Side/Dog)
(bar.f <- findbars(mform)) # list with 3 terms
mf <- model.frame(subbars(mform),data=Pixel)
rt <- mkReTrms(bar.f,mf)
names(rt)



cleanEx()
nameEx("modular")
### * modular

flush(stderr()); flush(stdout())

### Name: modular
### Title: Modular Functions for Mixed Model Fits
### Aliases: glFormula lFormula mkGlmerDevfun mkLmerDevfun modular
###   optimizeGlmer optimizeLmer updateGlmerDevfun
### Keywords: models

### ** Examples

### Fitting a linear mixed model in 4 modularized steps

## 1.  Parse the data and formula:
lmod <- lFormula(Reaction ~ Days + (Days|Subject), sleepstudy)
names(lmod)
## 2.  Create the deviance function to be optimized:
(devfun <- do.call(mkLmerDevfun, lmod))
ls(environment(devfun)) # the environment of 'devfun' contains objects
                        # required for its evaluation
## 3.  Optimize the deviance function:
opt <- optimizeLmer(devfun)
opt[1:3]
## 4.  Package up the results:
mkMerMod(environment(devfun), opt, lmod$reTrms, fr = lmod$fr)


### Same model in one line
lmer(Reaction ~ Days + (Days|Subject), sleepstudy)


### Fitting a generalized linear mixed model in six modularized steps

## 1.  Parse the data and formula:
glmod <- glFormula(cbind(incidence, size - incidence) ~ period + (1 | herd),
                   data = cbpp, family = binomial)
    #.... see what've got :
str(glmod, max=1, give.attr=FALSE)
## 2.  Create the deviance function for optimizing over theta:
(devfun <- do.call(mkGlmerDevfun, glmod))
ls(environment(devfun)) # the environment of devfun contains lots of info
## 3.  Optimize over theta using a rough approximation (i.e. nAGQ = 0):
(opt <- optimizeGlmer(devfun))
## 4.  Update the deviance function for optimizing over theta and beta:
(devfun <- updateGlmerDevfun(devfun, glmod$reTrms))
## 5.  Optimize over theta and beta:
opt <- optimizeGlmer(devfun, stage=2)
str(opt, max=1) # seeing what we'got
## 6.  Package up the results:
(fMod <- mkMerMod(environment(devfun), opt, glmod$reTrms, fr = glmod$fr))

### Same model in one line
fM <- glmer(cbind(incidence, size - incidence) ~ period + (1 | herd),
            data = cbpp, family = binomial)
all.equal(fMod, fM, check.attributes=FALSE, tolerance = 1e-12)
        # ----  --  even tolerance = 0  may work



cleanEx()
nameEx("namedList")
### * namedList

flush(stderr()); flush(stdout())

### Name: namedList
### Title: Self-naming list function
### Aliases: namedList

### ** Examples

a <- 1
b <- 2
c <- 3
str(namedList(a, b, d = c))



cleanEx()
nameEx("ngrps")
### * ngrps

flush(stderr()); flush(stdout())

### Name: ngrps
### Title: Number of Levels of a Factor or a "merMod" Model
### Aliases: ngrps

### ** Examples

ngrps(factor(seq(1,10,2)))
ngrps(lmer(Reaction ~ 1|Subject, sleepstudy))

## A named vector if there's more than one grouping factor :
ngrps(lmer(strength ~ (1|batch/cask), Pastes))
## cask:batch      batch
##         30         10

methods(ngrps) # -> "factor" and "merMod"



cleanEx()
nameEx("nlmer")
### * nlmer

flush(stderr()); flush(stdout())

### Name: nlmer
### Title: Fitting Nonlinear Mixed-Effects Models
### Aliases: nlmer
### Keywords: models

### ** Examples

## nonlinear mixed models --- 3-part formulas ---
## 1. basic nonlinear fit. Use stats::SSlogis for its
## implementation of the 3-parameter logistic curve.
## "SS" stands for "self-starting logistic", but the
## "self-starting" part is not currently used by nlmer ... 'start' is
## necessary
startvec <- c(Asym = 200, xmid = 725, scal = 350)
(nm1 <- nlmer(circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym|Tree,
             Orange, start = startvec))
## 2. re-run with "quick and dirty" PIRLS step
(nm1a <- update(nm1, nAGQ = 0L))

## 3. Fit the same model with a user-built function:
## a. Define formula
nform <- ~Asym/(1+exp((xmid-input)/scal))
## b. Use deriv() to construct function:
nfun <- deriv(nform,namevec=c("Asym","xmid","scal"),
              function.arg=c("input","Asym","xmid","scal"))
nm1b <- update(nm1,circumference ~ nfun(age, Asym, xmid, scal)  ~ Asym | Tree)

## 4. User-built function without using deriv():
##    derivatives could be computed more efficiently
##    by pre-computing components, but these are essentially
##    the gradients as one would derive them by hand
nfun2 <- function(input, Asym, xmid, scal) {
    value <- Asym/(1+exp((xmid-input)/scal))
    grad <- cbind(Asym=1/(1+exp((xmid-input)/scal)),
              xmid=-Asym/(1+exp((xmid-input)/scal))^2*1/scal*
                    exp((xmid-input)/scal),
              scal=-Asym/(1+exp((xmid-input)/scal))^2*
                     -(xmid-input)/scal^2*exp((xmid-input)/scal))
    attr(value,"gradient") <- grad
    value
}
stopifnot(all.equal(attr(nfun(2,1,3,4),"gradient"),
                    attr(nfun(2,1,3,4),"gradient")))
nm1c <- update(nm1,circumference ~ nfun2(age, Asym, xmid, scal)  ~ Asym | Tree)



cleanEx()
nameEx("nloptwrap")
### * nloptwrap

flush(stderr()); flush(stdout())

### Name: nloptwrap
### Title: Wrappers for additional optimizers
### Aliases: nloptwrap nlminbwrap

### ** Examples

library(lme4)
ls.str(environment(nloptwrap)) # 'defaultControl' algorithm "NLOPT_LN_BOBYQA"
## Note that  'optimizer =  "nloptwrap"' is now the default for lmer() :
fm1 <- lmer(Reaction ~ Days + (Days|Subject), sleepstudy)
## tighten tolerances
fm1B <- update(fm1, control= lmerControl(optCtrl = list(xtol_abs=1e-8, ftol_abs=1e-8)))
## run for longer (no effect in this case)
fm1C <- update(fm1,control = lmerControl(optCtrl = list(maxeval=10000)))

  logLik(fm1B) - logLik(fm1)  ## small difference in log likelihood
c(logLik(fm1C) - logLik(fm1)) ## no difference in LL
## Nelder-Mead
fm1_nloptr_NM <- update(fm1, control=
                  lmerControl(optimizer = "nloptwrap",
                              optCtrl = list(algorithm = "NLOPT_LN_NELDERMEAD")))
## other nlOpt algorithm options include NLOPT_LN_COBYLA, NLOPT_LN_SBPLX, see
if(interactive())
  nloptr::nloptr.print.options()

fm1_nlminb <- update(fm1, control=lmerControl(optimizer = "nlminbwrap"))
if (require(optimx)) { ## the 'optimx'-based nlminb :
  fm1_nlminb2 <- update(fm1, control=
                lmerControl(optimizer = "optimx",
                            optCtrl = list(method="nlminb", kkt=FALSE)))
  cat("Likelihood difference (typically zero):  ",
      c(logLik(fm1_nlminb) - logLik(fm1_nlminb2)), "\n")
}





cleanEx()
nameEx("nobars")
### * nobars

flush(stderr()); flush(stdout())

### Name: nobars
### Title: Omit terms separated by vertical bars in a formula
### Aliases: nobars
### Keywords: models utilities

### ** Examples

nobars(Reaction ~ Days + (Days|Subject)) ## => Reaction ~ Days



cleanEx()
nameEx("plot.merMod")
### * plot.merMod

flush(stderr()); flush(stdout())

### Name: plot.merMod
### Title: Diagnostic Plots for 'merMod' Fits
### Aliases: plot.merMod qqmath.merMod

### ** Examples

data(Orthodont,package="nlme")
fm1 <- lmer(distance ~ age + (age|Subject), data=Orthodont)
## standardized residuals versus fitted values by gender
plot(fm1, resid(., scaled=TRUE) ~ fitted(.) | Sex, abline = 0)
## box-plots of residuals by Subject
plot(fm1, Subject ~ resid(., scaled=TRUE))
## observed versus fitted values by Subject
plot(fm1, distance ~ fitted(.) | Subject, abline = c(0,1))
## residuals by age, separated by Subject
plot(fm1, resid(., scaled=TRUE) ~ age | Sex, abline = 0)
## scale-location plot, with red smoothed line
scale_loc_plot <- function(m, line.col = "red", line.lty = 1,
                           line.lwd = 2) {
  plot(fm1, sqrt(abs(resid(.))) ~ fitted(.),
       type = c("p", "smooth"),
       par.settings = list(plot.line =
                             list(alpha=1, col = line.col,
                                  lty = line.lty, lwd = line.lwd)))
}
scale_loc_plot(fm1)
## Q-Q plot
lattice::qqmath(fm1, id=0.05)
ggp.there <- "package:ggplot2" %in% search()
if (ggp.there || require("ggplot2")) {
    ## we can create the same plots using ggplot2 and the fortify() function
    fm1F <- fortify.merMod(fm1)
    ggplot(fm1F, aes(.fitted, .resid)) + geom_point(colour="blue") +
           facet_grid(. ~ Sex) + geom_hline(yintercept=0)
    ## note: Subjects are ordered by mean distance
    ggplot(fm1F, aes(Subject,.resid)) + geom_boxplot() + coord_flip()
    ggplot(fm1F, aes(.fitted,distance)) + geom_point(colour="blue") +
        facet_wrap(~Subject) +geom_abline(intercept=0,slope=1)
    ggplot(fm1F, aes(age,.resid)) + geom_point(colour="blue") + facet_grid(.~Sex) +
        geom_hline(yintercept=0)+ geom_line(aes(group=Subject),alpha=0.4) +
        geom_smooth(method="loess")
    ## (warnings about loess are due to having only 4 unique x values)
    if(!ggp.there) detach("package:ggplot2")
}



cleanEx()
nameEx("plots.thpr")
### * plots.thpr

flush(stderr()); flush(stdout())

### Name: plots.thpr
### Title: Mixed-Effects Profile Plots (Regular / Density / Pairs)
### Aliases: xyplot.thpr densityplot.thpr splom.thpr

### ** Examples

## see   example("profile.merMod")



cleanEx()
nameEx("predict.merMod")
### * predict.merMod

flush(stderr()); flush(stdout())

### Name: predict.merMod
### Title: Predictions from a model at new data values
### Aliases: predict.merMod

### ** Examples

(gm1 <- glmer(cbind(incidence, size - incidence) ~ period + (1 |herd), cbpp, binomial))
str(p0 <- predict(gm1))            # fitted values
str(p1 <- predict(gm1,re.form=NA))  # fitted values, unconditional (level-0)
newdata <- with(cbpp, expand.grid(period=unique(period), herd=unique(herd)))
str(p2 <- predict(gm1,newdata))    # new data, all RE
str(p3 <- predict(gm1,newdata,re.form=NA)) # new data, level-0
str(p4 <- predict(gm1,newdata,re.form= ~(1|herd))) # explicitly specify RE
stopifnot(identical(p2, p4))
## Don't show: 

## predict() should work with variable names with spaces [as lm() does]:
dd <- expand.grid(y=1:3, "Animal ID" = 1:9)
fm <- lmer(y ~ 1 + (1 | `Animal ID`),  dd)
summary(fm)
isel <- c(7, 9, 11, 13:17, 20:22)
stopifnot(all.equal(vcov(fm)[1,1], 0.02564102564),
	  all.equal(unname(predict(fm, newdata = dd[isel,])),
		    unname( fitted(fm) [isel])))
## End(Don't show)
 



cleanEx()
nameEx("profile-methods")
### * profile-methods

flush(stderr()); flush(stdout())

### Name: profile-methods
### Title: Profile method for merMod objects
### Aliases: as.data.frame.thpr log.thpr logProf varianceProf
###   profile-methods profile.merMod
### Keywords: methods

### ** Examples

fm01ML <- lmer(Yield ~ 1|Batch, Dyestuff, REML = FALSE)
system.time(
  tpr  <- profile(fm01ML, optimizer="Nelder_Mead", which="beta_")
)## fast; as only *one* beta parameter is profiled over -> 0.09s (2022)

## full profiling (default which means 'all') needs longer:
system.time( tpr  <- profile(fm01ML, signames=FALSE))
## ~ 0.26s (2022) + possible warning about convergence
(confint(tpr) -> CIpr)
if (interactive()) {
 library("lattice")
 xyplot(tpr)
 xyplot(tpr, absVal=TRUE) # easier to see conf.int.s (and check symmetry)
 xyplot(tpr, conf = c(0.95, 0.99), # (instead of all five 50, 80,...)
        main = "95% and 99% profile() intervals")
 xyplot(logProf(tpr, ranef=FALSE),
        main = expression("lmer profile()s"~~ log(sigma)*" (only log)"))
 densityplot(tpr, main="densityplot( profile(lmer(..)) )")
 densityplot(varianceProf(tpr), main=" varianceProf( profile(lmer(..)) )")
 splom(tpr)
 splom(logProf(tpr, ranef=FALSE))
 doMore <- lme4:::testLevel() > 2 
 if(doMore) { ## not typically, for time constraint reasons
   ## Batch and residual variance only
   system.time(tpr2 <- profile(fm01ML, which=1:2)) # , optimizer="Nelder_Mead" gives warning
   print( xyplot(tpr2) )
   print( xyplot(log(tpr2)) )# log(sigma) is better
   print( xyplot(logProf(tpr2, ranef=FALSE)) )

   ## GLMM example
   gm1 <- glmer(cbind(incidence, size - incidence) ~ period + (1 | herd),
               data = cbpp, family = binomial)
   ## running ~ 10 seconds on a modern machine {-> "verbose" while you wait}:
   print( system.time(pr4 <- profile(gm1, verbose=TRUE)) )
   print( xyplot(pr4, layout=c(5,1), as.table=TRUE) )
   print( xyplot(log(pr4), absVal=TRUE) ) # log(sigma_1)
   print( splom(pr4) )
   print( system.time( # quicker: only sig01 and one fixed effect
       pr2 <- profile(gm1, which=c("theta_", "period2"))))
   print( confint(pr2) )
   ## delta..: higher underlying resolution, only for 'sigma_1':
   print( system.time(
       pr4.hr <- profile(gm1, which="theta_", delta.cutoff=1/16)))
   print( xyplot(pr4.hr) )
 }
} # only if interactive()



cleanEx()
nameEx("ranef")
### * ranef

flush(stderr()); flush(stdout())

### Name: ranef
### Title: Extract the modes of the random effects
### Aliases: ranef ranef.merMod dotplot.ranef.mer qqmath.ranef.mer
###   as.data.frame.ranef.mer
### Keywords: methods models

### ** Examples

library(lattice) ## for dotplot, qqmath
fm1 <- lmer(Reaction ~ Days + (Days|Subject), sleepstudy)
fm2 <- lmer(Reaction ~ Days + (1|Subject) + (0+Days|Subject), sleepstudy)
fm3 <- lmer(diameter ~ (1|plate) + (1|sample), Penicillin)
ranef(fm1)
str(rr1 <- ranef(fm1))
dotplot(rr1)  ## default
qqmath(rr1)
## specify free scales in order to make Day effects more visible
dotplot(rr1,scales = list(x = list(relation = 'free')))[["Subject"]]
## plot options: ... can specify appearance of vertical lines with
## lty.v, col.line.v, lwd.v, etc..
dotplot(rr1, lty = 3, lty.v = 2, col.line.v = "purple",
        col = "red", col.line.h = "gray")
ranef(fm2)
op <- options(digits = 4)
ranef(fm3, drop = TRUE)
options(op)
## as.data.frame() provides RE's and conditional standard deviations:
str(dd <- as.data.frame(rr1))
if (require(ggplot2)) {
    ggplot(dd, aes(y=grp,x=condval)) +
        geom_point() + facet_wrap(~term,scales="free_x") +
        geom_errorbarh(aes(xmin=condval -2*condsd,
                           xmax=condval +2*condsd), height=0)
}



cleanEx()
nameEx("rePCA")
### * rePCA

flush(stderr()); flush(stdout())

### Name: rePCA
### Title: PCA of random-effects covariance matrix
### Aliases: rePCA

### ** Examples

  fm1 <- lmer(Reaction~Days+(Days|Subject), sleepstudy)
  rePCA(fm1)



cleanEx()
nameEx("rePos-class")
### * rePos-class

flush(stderr()); flush(stdout())

### Name: rePos-class
### Title: Class '"rePos"'
### Aliases: rePos-class
### Keywords: classes

### ** Examples

showClass("rePos")



cleanEx()
nameEx("refit")
### * refit

flush(stderr()); flush(stdout())

### Name: refit
### Title: Refit a (merMod) Model with a Different Response
### Aliases: refit refit.merMod

### ** Examples

## Ex. 1: using refit() to fit each column in a matrix of responses -------
set.seed(101)
Y <- matrix(rnorm(1000),ncol=10)
## combine first column of responses with predictor variables
d <- data.frame(y=Y[,1],x=rnorm(100),f=rep(1:10,10))
## (use check.conv.grad="ignore" to disable convergence checks because we
##  are using a fake example)
## fit first response
fit1 <- lmer(y ~ x+(1|f), data = d,
             control= lmerControl(check.conv.grad="ignore",
                                  check.conv.hess="ignore"))
## combine fit to first response with fits to remaining responses
res <- c(fit1, lapply(as.data.frame(Y[,-1]), refit, object=fit1))

## Ex. 2: refitting simulated data using data that contain NA values ------
sleepstudyNA <- sleepstudy
sleepstudyNA$Reaction[1:3] <- NA
fm0 <- lmer(Reaction ~ Days + (1|Subject), sleepstudyNA)
## the special case of refitting with a single simulation works ...
ss0 <- refit(fm0, simulate(fm0))
## ... but if simulating multiple responses (for efficiency),
## need to use na.action=na.exclude in order to have proper length of data
fm1 <- lmer(Reaction ~ Days + (1|Subject), sleepstudyNA, na.action=na.exclude)
ss <- simulate(fm1, 5)
res2 <- refit(fm1, ss[[5]])



cleanEx()
nameEx("sigma")
### * sigma

flush(stderr()); flush(stdout())

### Name: sigma
### Title: Extract Residual Standard Deviation 'Sigma'
### Aliases: sigma sigma.merMod

### ** Examples

methods(sigma)# from R 3.3.0 on, shows methods from pkgs 'stats' *and* 'lme4'



cleanEx()
nameEx("simulate.merMod")
### * simulate.merMod

flush(stderr()); flush(stdout())

### Name: simulate.merMod
### Title: Simulate Responses From 'merMod' Object
### Aliases: simulate.merMod .simulateFun

### ** Examples

## test whether fitted models are consistent with the
##  observed number of zeros in CBPP data set:
gm1 <- glmer(cbind(incidence, size - incidence) ~ period + (1 | herd),
             data = cbpp, family = binomial)
gg <- simulate(gm1,1000)
zeros <- sapply(gg,function(x) sum(x[,"incidence"]==0))
plot(table(zeros))
abline(v=sum(cbpp$incidence==0),col=2)
##
## simulate from a non-fitted model; in this case we are just
## replicating the previous model, but starting from scratch
params <- list(theta=0.5,beta=c(2,-1,-2,-3))
simdat <- with(cbpp,expand.grid(herd=levels(herd),period=factor(1:4)))
simdat$size <- 15
simdat$incidence <- sample(0:1,size=nrow(simdat),replace=TRUE)
form <- formula(gm1)[-2]  ## RHS of equation only
simulate(form,newdata=simdat,family=binomial,
    newparams=params)
## simulate from negative binomial distribution instead
simulate(form,newdata=simdat,family=negative.binomial(theta=2.5),
    newparams=params)



cleanEx()
nameEx("sleepstudy")
### * sleepstudy

flush(stderr()); flush(stdout())

### Name: sleepstudy
### Title: Reaction times in a sleep deprivation study
### Aliases: sleepstudy
### Keywords: datasets

### ** Examples

str(sleepstudy)
require(lattice)
xyplot(Reaction ~ Days | Subject, sleepstudy, type = c("g","p","r"),
       index = function(x,y) coef(lm(y ~ x))[1],
       xlab = "Days of sleep deprivation",
       ylab = "Average reaction time (ms)", aspect = "xy")
(fm1 <- lmer(Reaction ~ Days + (Days|Subject), sleepstudy, subset=Days>=2))
## independent model
(fm2 <- lmer(Reaction ~ Days + (1|Subject) + (0+Days|Subject), sleepstudy, subset=Days>=2))



cleanEx()
nameEx("subbars")
### * subbars

flush(stderr()); flush(stdout())

### Name: subbars
### Title: "Sub[stitute] Bars"
### Aliases: subbars
### Keywords: models utilities

### ** Examples

subbars(Reaction ~ Days + (Days|Subject)) ## => Reaction ~ Days + (Days + Subject)



cleanEx()
nameEx("utilities")
### * utilities

flush(stderr()); flush(stdout())

### Name: prt-utilities
### Title: Print and Summary Method Utilities for Mixed Effects
### Aliases: .prt.methTit .prt.VC .prt.aictab .prt.call .prt.family
###   .prt.grps .prt.methTit .prt.resids .prt.warn formatVC llikAIC
###   methTitle
### Keywords: utilities

### ** Examples

## Create a few "lme4 standard" models ------------------------------
fm1 <- lmer(Reaction ~ Days + (Days | Subject), sleepstudy)
fmM <- update(fm1, REML=FALSE) # -> Maximum Likelihood
fmQ <- update(fm1, . ~ Days + (Days | Subject))

gm1 <- glmer(cbind(incidence, size - incidence) ~ period + (1 | herd),
             data = cbpp, family = binomial)
gmA <- update(gm1, nAGQ = 5)


(lA1 <- llikAIC(fm1))
(lAM <- llikAIC(fmM))
(lAg <- llikAIC(gmA))

(m1 <- methTitle(fm1 @ devcomp $ dims))
(mM <- methTitle(fmM @ devcomp $ dims))
(mG <- methTitle(gm1 @ devcomp $ dims))
(mA <- methTitle(gmA @ devcomp $ dims))

.prt.methTit(m1, class(fm1))
.prt.methTit(mA, class(gmA))

.prt.family(gaussian())
.prt.family(binomial())
.prt.family( poisson())

.prt.resids(residuals(fm1), digits = 4)
.prt.resids(residuals(fmM), digits = 2)

.prt.call(getCall(fm1))
.prt.call(getCall(gm1))

.prt.aictab ( lA1 $ AICtab ) # REML
.prt.aictab ( lAM $ AICtab ) # ML --> AIC, BIC, ...

V1 <- VarCorr(fm1)
m <- formatVC(V1)
stopifnot(is.matrix(m), is.character(m), ncol(m) == 4)
print(m, quote = FALSE) ## prints all but the first line of .prt.VC() below:
.prt.VC( V1, digits = 4)
## Random effects:
##  Groups   Name        Std.Dev. Corr
##  Subject  (Intercept) 24.740
##           Days         5.922   0.07
##  Residual             25.592
p1 <- capture.output(V1)
p2 <- capture.output( print(m, quote=FALSE) )
pX <- capture.output( .prt.VC(V1, digits = max(3, getOption("digits")-2)) )
stopifnot(identical(p1, p2),
          identical(p1, pX[-1])) # [-1] : dropping 1st line

(Vq <- VarCorr(fmQ)) # default print()
print(Vq, comp = c("Std.Dev.", "Variance"))
print(Vq, comp = c("Std.Dev.", "Variance"), corr=FALSE)
print(Vq, comp = "Variance")

.prt.grps(ngrps = ngrps(fm1),
          nobs  = nobs (fm1))
## --> Number of obs: 180, groups:  Subject, 18

.prt.warn(fm1 @ optinfo) # nothing .. had no warnings
.prt.warn(fmQ @ optinfo) # (ditto)



cleanEx()
nameEx("vcconv")
### * vcconv

flush(stderr()); flush(stdout())

### Name: vcconv
### Title: Convert between representations of (co-)variance structures
### Aliases: vcconv mlist2vec vec2mlist vec2STlist sdcor2cov cov2sdcor
###   Vv_to_Cv Sv_to_Cv Cv_to_Vv Cv_to_Sv

### ** Examples

vec2mlist(1:6)
mlist2vec(vec2mlist(1:6)) # approximate inverse



cleanEx()
nameEx("vcov.merMod")
### * vcov.merMod

flush(stderr()); flush(stdout())

### Name: vcov.merMod
### Title: Covariance matrix of estimated parameters
### Aliases: vcov.merMod vcov.summary.merMod
### Keywords: models

### ** Examples

fm1 <- lmer(Reaction ~ Days + (Days | Subject), sleepstudy)
gm1 <- glmer(cbind(incidence, size - incidence) ~ period + (1 | herd),
             data = cbpp, family = binomial)
(v1 <- vcov(fm1))
v2 <- vcov(fm1, correlation = TRUE)
# extract the hidden 'correlation' entry in @factors
as(v2, "corMatrix")
v3 <- vcov(gm1)
v3X <- vcov(gm1, use.hessian  = FALSE)
all.equal(v3, v3X)
## full correlatiom matrix
cv <- vcov(fm1, full = TRUE)
image(cv, xlab = "", ylab = "",
      scales = list(y = list(labels = rownames(cv)),
                    at = seq(nrow(cv)),
                    x = list(labels = NULL)))



### * <FOOTER>
###
cleanEx()
options(digits = 7L)
base::cat("Time elapsed: ", proc.time() - base::get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')
