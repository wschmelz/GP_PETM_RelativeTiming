# GP_PETM_RelativeTiming
This repository, [GP_PETM_RelativeTiming](https://github.com/wschmelz/GP_PETM_RelativeTiming.git), contains the code to run the Gaussian Process regression analyses documented by Makarova et al. (2025) in the _Paleoceanography and Paleoclimatology_ article titled "Warming and Carbon Injection at the Paleocene-Eocene Boundary: Bayesian Modeling Supports Synchroneity".

# Short methods
We applied a Bayesian hierarchical model that utilizes Gaussian Process priors to estimate variations in d13C and TEXH86​ data from: 1) two cores drilled 2.4 m apart at Medford Auger Project (MAP) site 3; and 2) a depth transect of five sites on the New Jersey Coastal Plain (NJCP) that includes Medford, Millville, Wilson Lake, Ancora, and Bass River. This statistical modeling produces posterior estimates of d13C and TEXH86​-dervied paleotemperature variations recorded at MAP site 3 and across the NJCP through the PETM. We use these statistically modeled estimates of the d13C and TEXH86​-dervied paleotemperature variations to evaluate the relative timing of the initation of the CIE and the initiation of the warming associated with the PETM. We estimate the point of initation of the CIE and the warming associated with the PETM by sampling functions defined by the posterior covariance matrix and calculating the rate of variation in d13C and TEXH86​-dervied paleotemperature with respect to depth/time through the stratigraphic sections.

# Use
To run the analyses documented in the manuscript, execute the "job_script.py" file that is located in the two directories within this repository, "00_Medford" and "01_NJCP". The code has been successfully tested with the Anaconda distribution of Python 3.9. This version of the code will run sucessfully using a Windows OS, if the necessary Python modules are installed.
