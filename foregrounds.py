import numpy as np
from numpy.linalg import *
import astropy.io.fits as fits
import scipy.ndimage as nd


#------------------------------------------------------------------------------------------------
#Foreground class
#------------------------------------------------------------------------------------------------
class fgmodel(object):
    '''
    Class of foreground model for the Hillipop likelihood
    Units: Dl in muK^2
    '''
    
    # MJy.sr-1 -> Kcmb for (100,143,217) GHz
    gnu = {100:244.059, 143:371.658, 217:483.485}
    #frequency dependence (100,143,217,353,545,857) GHz of the SZ effect
    fnu = {100:-4.031, 143:-2.785, 217:0.187, 353:6.205, 545:14.455, 857:26.335}

    def __init__( self, freqs, parname, filename, auto=False):
        '''
        Create model for foreground
        '''
        self.freq = freqs
        self.name = "MyFg"
        parname.append( "par1")
        pass
    
    def compute_dl( self, pars):
        '''
        Return spectra model for each cross-frequency
        '''
        pass

#------------------------------------------------------------------------------------------------
