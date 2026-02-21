
import numpy as np
import fitsutils




def norm_method(arr, point):
    point = np.asarray(point)

    idx = np.indices(arr.shape, sparse=True)
    idx = (idx[0] - point[0], idx[1]-point[1])
    # new numpy requires object dtype for ragged lists
    a = np.array([idx[0], idx[1]], dtype=object)

    norm = np.linalg.norm(a)
    
    return norm

def get_source_pixels_in_aperture(ra, dec, radius, fitsobj):
    scale = int((radius / abs(fitsobj.cdelt1)))
    xr, yr = fitsobj.world2pix(ra, dec)
    dist = norm_method(fitsobj.data, (xr, yr))

    cond1 = dist <= scale
    indices_inside = np.where(cond1)

    return indices_inside



def extract_flux_density(fitsobj, indices_x, indices_y, sigma=-1000):
    """
    """

    source_flux = fitsobj.data[indices_x, indices_y].flatten()
    source_bpp = fitsobj.bpp[indices_x, indices_y].flatten()
    source_solid_angle = fitsobj.solid_angle[indices_x, indices_y].flatten()
    source_sb = source_flux/(source_solid_angle*3600.*3600.)

    source_rms = fitsobj.rms[indices_x, indices_y].flatten()

    source_rms_full = source_rms.copy()
    source_bpp_full = source_bpp.copy()

    # n_good_pixels = len(np.where(source_flux >= source_rms*sigma)[0])
    idx = np.where(source_flux < source_rms*sigma)
    
    source_bpp[idx] = np.nan
    source_rms[idx] = np.nan
    source_sb[idx] = np.nan
    source_flux[idx] = np.nan
    



    source_int_flux = np.nansum(source_flux / source_bpp)
    source_peak_flux = np.nanmax(source_flux)

    source_unc_flux = (np.nansum(source_rms_full / source_bpp_full) * \
        np.sqrt(np.nanmean(source_bpp_full) / float(len(source_rms_full))))
    source_rms_avg = np.nanmean(source_rms_full)
        

    params = [
        source_int_flux,  # 0
        source_peak_flux, # 1
        source_unc_flux,  # 2
        source_rms_avg,   # 3
    ]
    params = [float(item) for item in params]
    
    return params


def measure_flux_density(fitsimage, rms, 
    coords=None,
    radius=None,    
    r_index=0, 
    sigma : float = -1000):
    """
    
    Inspired by `radioflux.py` by
    Martin Hardcastle: https://github.com/mhardcastle/radioflux


    
    """

    fitsobj = fitsutils.Fits(fitsimage)
    fitsobj.add_beam(
        bmaj=None,
        bmin=None,
        psfimage=None
    ) 
    fitsobj.add_rms(rms=rms) 
    fitsobj.add_beams_per_pixel()

    # get source pixels
    indices_x, indices_y = get_source_pixels_in_aperture(
        ra=coords[0], 
        dec=coords[1],
        radius=radius, 
        fitsobj=fitsobj
        )
    
    params = extract_flux_density(
        fitsobj=fitsobj, 
        indices_x=indices_x, 
        indices_y=indices_y, 
        sigma=sigma
    )

    print("Region {}: flux density       = {} ({}) Jy".format(r_index, params[0], params[2]))
    print("Region {}: peak flux (rms)    = {} ({}) Jy/beam".format(r_index, params[1], params[3]))


    return params

