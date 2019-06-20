# Changelog
All notable changes to LIBTwinSVM library will be documented in this file. The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2019-06-17
### Added
- Added two benchmark scripts to the repository.
- Added Grid Search method to the Library's API.
- Added two more usage examples to the documentation of the project.

### Fixed
- Fixed adding LAPACK and BLAS DLLs for Windows platform.
- Fixed an Installation error for Linux systems.
- A workaround for installing the library on MacOSX systems.
- To overcome matrix singularity of linear LSTSVM, an extra stabilizer term added to the equations.

## [0.1.0] - 2019-05-10
### Added
- A graphical user interface (GUI) application.
- Fast implementation of standard TwinSVM and Least Sqaures TwinSVM.
- A fast optimizer (ClipDCD) in Cython.
- Implemented multi-class schemes (One-vs-All & One-vs-One) as meta-estimators.
- A feature-rich visualization tool to show decision boundaries.
- A module for saving and loading TSVM-based models.
- A setup.py file for installing LIBTwinSVM as a Python package.
