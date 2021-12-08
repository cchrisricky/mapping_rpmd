#Module which includes the functions dealing with the normal modes of the ring-polymer

import numpy as np

###############################################################

def calc_normal_mode_freq( beta_p, nbds ):

    #Calculate frequencies of normal modes of ring-polymer
    #note that these are mass-independent and are thus the same for all nuclei

    nm = 2.0 / beta_p * np.sin( np.arange(nbds) * np.pi / nbds )

    return nm

###############################################################

def real_to_normal_mode( real_space ):

    #Takes input array and calculates normal-modes using fourier-transform
    #Assumes input array is real and transforms complex output from FT to
    #real-valued normal modes by taking sums of real and imaginary components of degenerate frequency modes
    
    #discrete fourier transform, which gives complex results
    nm_cmplx = np.fft.rfft( real_space, norm='ortho' )
    
    #intialize terms for real valued normal mode array
    sz   = real_space.size
    ndeg = int( np.ceil( sz/2 ) -1 ) #number of degenerate frequencies
    nm   = np.zeros(sz)
    
    #define real valued normal modes
    nm[0] = np.real( nm_cmplx[0] ) #0-frequency mode, always real
    nm[1:ndeg+1] = np.real( np.sqrt(0.5) * ( nm_cmplx[1:ndeg+1] + np.conjugate(nm_cmplx[1:ndeg+1]) ) )
    if( sz % 2 == 0 ):
        nm[ndeg+1]  = np.real( nm_cmplx[-1] ) #other non-degenerate real mode if even number of beads
        nm[ndeg+2:] = np.flip( np.imag( np.sqrt(0.5) * ( nm_cmplx[1:ndeg+1] - np.conjugate(nm_cmplx[1:ndeg+1]) ) ) )
    else:
        nm[ndeg+1:] = np.flip( np.imag( np.sqrt(0.5) * ( nm_cmplx[1:ndeg+1] - np.conjugate(nm_cmplx[1:ndeg+1]) ) ) )

    return nm

###############################################################

def normal_mode_to_real( nm ):

    #Takes input array of real-valued normal modes and calculates inverse FT
    #to obtain array in real-space

    #initialize terms for complex valued normal mode array
    sz         = nm.size
    ndeg       = int( np.ceil( sz/2 ) -1 ) #number of degenerate frequencies
    if( sz % 2 == 0 ):
        nm_cmplx = np.zeros(ndeg+2, dtype=complex) #even bead number has additional non-degenerate mode
    else:
        nm_cmplx = np.zeros(ndeg+1, dtype=complex)

    #convert real valued to complex valued normal modes
    nm_cmplx[0] = nm[0]
    if( sz % 2 == 0 ):
        nm_cmplx[-1]       = nm[ndeg+1]
        nm_cmplx[1:ndeg+1] = np.sqrt(0.5) * ( nm[1:ndeg+1] + 1j * np.flip(nm[ndeg+2:]) )
    else:
        nm_cmplx[1:ndeg+1] = np.sqrt(0.5) * ( nm[1:ndeg+1] + 1j * np.flip(nm[ndeg+1:]) )

    #inverse fourier transform back to real space
    return np.fft.irfft( nm_cmplx, n=sz, norm='ortho' )

###############################################################
