# CASDA_query

If you make use of this of code for a publication, please acknowledge this repo.


### Notes/updates
Included an example for PSRJ1837–0616

Optimized both fatching of urls using the casda.cutout module and downloading of files.
Even for sources with over 100 ASKAP observation, the cutout/catalogue downloading now finishes in 10-15 mins (when requestion both Stokes I and Stokes V).
TO DO: optimize the batch sizes.

---

### What is this?

The Jupyter notebook in this repo allows users to query all ASKAP observations for a given coordinate by downloading source catalogues and image cutouts from CASDA. These products are then used this to build light curves (including flux values obtained by forced-fitting) and pngs of cutouts.

The tools directory hides the scripts that interact with the CASDA API to perform queries. These interactions can be slow for a large number of files, and have therefore been parallelized. 

The forced-fitting of images allows users to find sub-threshold detections. The forced-fitting code is adapted from the [radiofluxtools](https://gitlab.com/Sunmish/radiofluxtools) repo.



