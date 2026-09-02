#! /usr/bin/env python

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning

# import logging
# logging.basicConfig(format="%(levelname)s (%(module)s): %(message)s")
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

import warnings
warnings.filterwarnings("ignore", category=AstropyWarning) 


class Fits(object):
    """A class to hold and find relevant FITS image information.

    Here we are generally only interested in 2-dimensional images, as we 
    are usually interested only in the continuum flux density of sources.

    """

    def __init__(self, fitsimage, extension=0):
        """Open and/or assign a FITS image a class.

        Parameters
        ----------
        fitsimage : str or fits.HDUList object
            Either a filepath or an already opened/created fits.HDUList.
        extension : int, optional
            Extension in the HDU to get header/data information. [Default 0]
        squeeze : bool, optional
            Select True if wanting to remove extra axes. [Default True]

        This initialises the Fits object. This class in general is used to make
        openining FITS files and getting relevant information for flux density
        measurements easier for code/users. At least it seems that way to me!

        """
        
        if isinstance(fitsimage, str):
            # Open FITS from file:
            hdu = fits.open(fitsimage)
            opened = True
        elif isinstance(fitsimage, fits.HDUList):
            # Already opened:
            hdu = fitsimage
            opened = False
        else:
            raise IOError("`fitsimage` must be one of: str or fits.HDUList.")

        # makes the find_freq stuff fail - best use find_freq here?
        # self.hdr = strip_wcsaxes(hdu[extension].header)
        self.hdr = hdu[extension].header
        self.data = np.squeeze(hdu[extension].data)
        self.wcs = WCS(hdu[extension].header).celestial
        self.filename = fitsimage

        if "CDELT1" in self.hdr.keys():
            self.cdelt1 = self.hdr["CDELT1"]
            self.cdelt2 = self.hdr["CDELT2"]
        elif "CD1_1" in self.hdr.keys():
            self.cdelt1 = self.hdr["CD1_1"]
            self.cdelt2 = self.hdr["CD2_2"]
        else:
            raise ValueError("Pixel sizes cannot be determined.")

        if opened:
            hdu.close()

        self.bmaj = None
        self.bmin = None



    def pix2world(self, x, y):
        """Wrapper for a wrapper of all_pix2world."""
        return pix_to_world(self.wcs, x, y)



    def world2pix(self, ra, dec, no_int=False):
        """Wrapper of a wrapper of all_world2pix."""
        x, y =  world_to_pix(self.wcs, ra, dec)
        # if not no_int:
        #     try:
        #         x = int(x)
        #         y = int(y)
        #     except TypeError:
        #         x = x.astype("i")
        #         y = y.astype("i")

        return x, y
    

    def add_beams_per_pixel(self):
        """Calculate the fraction of the beam a pixel occupies.

        Used for calculation of integrated flux density over a number of pixels.
        """

        if self.bmaj is None:
            self.add_beam()
        self.bpp = self.solid_angle / abs(self.cdelt1*self.cdelt2)


    def add_beam(self, bmaj=None, bmin=None, psfimage=None):
        """Add beam information manually."""

        if psfimage is None and bmaj is None:
            self.find_beam()
            self.bmaj = np.full_like(self.data, self.bmaj)
            self.bmin = np.full_like(self.data, self.bmin)
        elif psfimage is not None:
            psfdata = fits.getdata(psfimage)
            self.bmaj = psfdata[0, :, :]
            self.bmin = psfdata[1, :, :]
        else:
            self.bmaj = np.full_like(self.data, bmaj)
            self.bmin = np.full_like(self.data, bmin)
        
        self.solid_angle = (np.pi * self.bmaj * self.bmin) / (4. * np.log(2.))



    def find_beam(self):
        """Find restoring/synthesized beam information."""

        if ("BMAJ" in self.hdr.keys()) and ("BMIN" in self.hdr.keys()):
        
            self.bmaj = self.hdr["BMAJ"]
            self.bmin = self.hdr["BMIN"]
        
        elif ("CLEANBMJ" in self.hdr.keys()) and ("CLEANBMN" in self.hdr.keys()):
        
            self.bmaj = self.hdr["CLEANBMJ"]
            self.bmin = self.hdr["CLEANBMN"]

        # Some specific surveys have specific beam functions:
        # Check for NVSS:
        elif  "nvss" in repr(self.hdr).lower():

            self.bmaj = 45./3600.
            self.bmin = 45./3600.

        # Check for SUMSS:
        elif "sumss" in repr(self.hdr).lower():

            self.bmaj = 45./3600.
            self.bmin = self.sumss_minor(self.hdr["CRVAL2"])/3600.

        # Check for TGSS:
        elif "tgss" in repr(self.hdr).lower():

            self.bmaj = 25./3600.
            self.bmin = self.tgss_minor(self.hdr["CRVAL2"])/3600.

        elif "bmaj" in repr(self.hdr).lower():
            pass

        if self.bmin is None:
            raise ValueError("No beam information found - try adding manually.")


    def find_freq(self):
        """Find frequency information in the FITS header.
        
        Check some normal locations."""

        freq = None
        if "FREQ" in self.hdr.keys():
            freq = self.hdr["FREQ"]
        for i in ["3", "4"]:
            if (freq is None) and ("CRVAL"+i in self.hdr.keys()):
                logger.debug(f"Checking CRVAL{i}")
                if "FREQ" in self.hdr["CTYPE"+i]:
                    freq = self.hdr["CRVAL"+i]
        if (freq is None) and ("CENTCHAN" in self.hdr.keys()):
            freq = 1.28*self.hdr["CENTCHAN"]*1.e6  # specific to MWA data

        if freq is None:
            raise ValueError("Cannot find frequency information for {}".format(self.filename))
    
        if freq < 1000.:
            # Likely in MHz, not really sensible to have < 1000 Hz?
            logger.warning("Assuming frequency in MHz for {}".format(self.filename))
            freq *= 1.e6

        self.freq = freq
        logger.debug("Found frequency {} MHz".format(self.freq/1e6))


    def add_rms(self, rms):
        """Add an rms array to the Fits object."""

        self.rms = None

        if rms is None:
            self.rms = np.full_like(self.data, np.nan)
        else:
            try:
                rms = float(rms)
                self.rms = np.full_like(self.data, rms)
            except ValueError:
                self.rms = np.squeeze(fits.getdata(rms))


    def writeout_source(self, indices_x, indices_y, outname):
        mask = self.data.copy()
        mask[indices_x, indices_y] = np.nan
        data = self.data.copy()
        data[~np.isnan(mask)] = np.nan
        fits.writeto(outname, data, self.hdr, overwrite=True)


    @staticmethod
    def sumss_minor(declination):
        """Calculate the SUMSS restoring beam for a given declination."""
        return 45./(np.sin(np.radians(declination)))

    @staticmethod
    def tgss_minor(declination):
        """Calculate the TGSS restoring beam for a given declination."""
        if declination >= 19.: 
            return 25.
        else:
            return 25./(np.cos(np.radians(declination - 19.)))


def ensure_array(a):
    """Ensure an object is an array, and if not make it so.

    Parameters
    ----------
    a : any
        This is turned into an array if not one already.

    Returns
    -------
    np.ndarray
        `a` as a np.ndarray.

    """


    if isinstance(a, np.ndarray) or isinstance(a, np.ma.MaskedArray):
        
        pass

    else:

        if hasattr(a, "__iter__") and not isinstance(a, str):
            a = np.asarray(a)
        elif isinstance(a, str):
            a = np.asarray([a])

    return a



def pix_to_world(wcs, x, y):
    """Convert from pixel to world coordinates.

    Parameters
    ----------
    x, y : int or arrays of ints
        x, y pixel coordinates.
    
    Returns
    -------
    array
        Two arrays: one of RA. one of dec. coordinates.

    """

    x, y = ensure_array(x), ensure_array(y)

    # We are only interested in the celestial coordinates:
    return wcs.celestial.all_pix2world(y, x, 0)

def world_to_pix(wcs, ra, dec):
    """Convert from world coordinates to pixel coordinates.

    Parameters
    ----------
    ra, dec : float or arrays of floats
        RA, dec. coordinates.

    Returns
    -------
    array
        Two int arrays for the x and y pixel coordinates.

    """
 
    ra, dec = ensure_array(ra), ensure_array(dec)

    # We are only interested in the celestial coordinates:
    y, x = wcs.celestial.all_world2pix(ra, dec, 0)

    # return x.astype("i"), y.astype("i")
    return x, y


def strip_wcsaxes(hdr):
    """Strip extra axes in a header object."""


    remove_keys = [key+i for key in 
                   ["CRVAL", "CDELT", "CRPIX", "CTYPE", "CUNIT", "NAXIS"]
                   for i in ["3", "4", "5"]]

    for key in remove_keys:
        if key in hdr.keys():
            del hdr[key]

    return hdr

def get_bpp(bmaj, bmin, cd1, cd2):
    return (np.pi*bmaj*bmin) / (4.*abs(cd1*cd2)*np.log(2.))
