---
# Easy
---
## Building and Checking the Package

The task involves writing a shell script to automate the process of installing dependencies, building the source tarball, and running `R CMD check` on the `lme4` package.

## **Introduction**

The `lme4` package is a widely used R package for fitting and analyzing linear, nonlinear and generalized linear mixed-models. The easy task involves:

1. Installing the dependencies of `lme4`.
2. Building the `lme4` source tarball.
3. Running `R CMD check` on the tarball.
4. Saving the check output`(lme4_easyTest_report)`.

This repository contains the shell script, check output, DESCRIPTION, lme4_easyTest_report.Rmd and lme4_easyTest_report.html file for the task.

---

### **Shell Script**

The shell script (`check_lme4.sh`) automates the process of building and checking the `lme4` package. It performs the following steps:

1. Installs the dependencies of `lme4`.
2. Builds the source tarball.
3. Runs `R CMD check` on the tarball.
4. Moves the check output to a `check_output` directory.
```
easy/
├── check_lme4.sh
├── check_output/
│   └── lme4.Rcheck/
│       ├── 00check.log
│       ├── 00install.out
│       ├── 00_pkg_src
│       ├── Rdlatex.log
│       ├── lme4-Ex.R
│       ├── lme4-Ex.Rout
│       ├── lme4-Ex.pdf
│       ├── lme4-manual.log
│       ├── lme4-manual.pdf
│       ├── lme4
│       └── tests
├── lme4_easyTest_report.Rmd
├── lme4_easyTest_report.html
└── DESCRIPTION

```
# Medium
Implements a diagonal random effects covariance matrix using `lme4` and compares results with `glmmTMB`.

### Observation

1.Diagonal covariance structure for random effects.

2.Fixed effects match between `lme4` and `glmmTMB`.

3.Random effects standard deviations are consistent.
