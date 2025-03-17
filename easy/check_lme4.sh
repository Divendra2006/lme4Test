#!/bin/bash

if [ ! -f "DESCRIPTION" ]; then
    echo "Error: This script must be run in the top-level directory of the lme4 source code."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
Rscript -e 'install.packages(c("Matrix", "methods", "stats", "graphics", "grid", "splines", "utils", "parallel", "MASS", "lattice", "boot", "nlme", "minqa", "nloptr", "reformulas", "Rcpp", "RcppEigen"), dependencies = TRUE)' || {
    echo "Failed to install dependencies. Exiting."
    exit 1
}

# Install suggested packages
echo "Installing suggested packages..."
Rscript -e 'install.packages(c("knitr", "rmarkdown", "testthat", "ggplot2", "mlmRev", "optimx", "gamm4", "pbkrtest", "HSAUR3", "numDeriv", "car", "dfoptim", "mgcv", "statmod", "rr2", "semEff", "tibble", "merDeriv"))' || {
    echo "Failed to install suggested packages. Exiting."
    exit 1
}

echo "Building the source tarball..."
R CMD build . || {
    echo "Failed to build the source tarball. Exiting."
    exit 1
}

TARBALL=$(ls lme4_*.tar.gz) || {
    echo "Failed to find the source tarball. Exiting."
    exit 1
}

echo "Checking the source tarball..."
R CMD check "${TARBALL}" || {
    echo "Failed to check the source tarball. Exiting."
    exit 1
}

echo "Moving check output to check_output directory..."
mkdir -p check_output
mv lme4.Rcheck check_output/ || {
    echo "Failed to move check output. Exiting."
    exit 1
}

echo "Done! Check output is in the check_output directory."