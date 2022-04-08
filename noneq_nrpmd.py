#Define class for non-equilibrium nrpmd
#It's a child-class of the map_rpmd parent class

import numpy as np
import utils
import map_rpmd
import nrpmd
import sys
import time
from scipy.linalg import expm

class noneq_nrpmd( map_rpmd.map_rpmd ):
    
    #####################################################################

    def __init__( self, nstates=1, nnuc=1, nbds=1, beta=1.0, mass=1.0, potype=None, potparams=None, mapR=None, mapP=None, nucR=None, nucP=None ):

        super().__init__( 'NRPMD', nstates, nnuc, nbds, beta, mass, potype, potparams, mapR, mapP, nucR, nucP )

        #initilaize Theta
        self.theta = None
        self.W     = None

    #####################################################################

    def get_timederivs( self ):
        #Subroutine to calculate the time-derivatives of all position/momenta

        #Update electronic Hamiltonian matrix
        self.potential.calc_Hel( self.nucR )

        #Calculate time-derivative of nuclear position
        dnucR = self.get_timederiv_nucR()

        #Calculate time-derivative of nuclear momenta
        dnucP = self.get_timederiv_nucP()

        #Calculate time-derivative of mapping position and momentum
        dmapR = self.get_timederiv_mapR()
        dmapP = self.get_timederiv_mapP()

        return dnucR, dnucP, dmapR, dmapP

    #####################################################################

    def get_theta( self ):
        #Subroutine to calculate Theta (minus the Gaussian term and prefactor) following Huo's 2019 paper https://dx.doi.org/10.1063/1.5096276

        #Electronic trace over the product of C matrices
        halfeye = 0.5 * np.eye(self.nstates)
        prod = np.outer(( self.mapR[0] + 1j*self.mapP[0] ), ( self.mapR[0] - 1j*self.mapP[0] )) - halfeye
        if self.nbds > 1:
            for i in range(1, self.nbds):
                prod = prod @ ( np.outer(( self.mapR[i] + 1j*self.mapP[i] ), ( self.mapR[i] - 1j*self.mapP[i] )) - halfeye )

        self.theta = np.real( np.trace( prod ) )

    #####################################################################

    def get_W( self ):
        #Subroutine to calculate W following Richardson's 2013 paper https://dx.doi.org/10.1063/1.4816124

        #Update electronic Hamiltonian matrix
        self.potential.calc_Hel( self.nucR )

        #M matrices
        Ms = np.zeros( [self.nbds, self.nstates, self.nstates] )
        for i in range( self.nbds ): Ms[i] = expm(-self.beta_p / 2 * self.potential.Hel[i])

        #Loop over each bead to calculate and multiply W
        self.W = 1.0
        for i in range( self.nbds ):
            self.W *= ( self.mapP[i - 1] @ Ms[i] @ self.mapR[i]) * ( self.mapR[i] @ Ms[i] @ self.mapP[i] )

    #####################################################################

    def get_timederiv_nucR( self ):
        #Subroutine to calculate the time-derivative of the nuclear positions for each bead

        return self.nucP / self.mass

    #####################################################################

    def get_timederiv_nucP( self, intRP_bool=True ):
        #Subroutine to calculate the time-derivative of the nuclear momenta for each bead

        #Force associated with harmonic springs between beads and the state-independent portion of the potential
        #This is dealt with in the parent class
        #If intRP_bool is False it does not calculate the contribution from the harmonic ring polymer springs

        if (self.nbds > 1):
            d_nucP = super().get_timederiv_nucP( intRP_bool )
        else:
            d_nucP = super().get_timederiv_nucP( intRP_bool = False )

        #Calculate nuclear derivative of electronic Hamiltonian matrix
        self.potential.calc_Hel_deriv( self.nucR )

        #Calculate contribution from MMST term
        #XXX could maybe make this faster getting rid of double index in einsum
        d_nucP += -0.5 * np.einsum( 'in,ijnm,im -> ij', self.mapR, self.potential.d_Hel, self.mapR )
        d_nucP += -0.5 * np.einsum( 'in,ijnm,im -> ij', self.mapP, self.potential.d_Hel, self.mapP )

        #add the state-average potential
        if (self.potype != 'harm_lin_cpl_symmetrized'):
            d_nucP +=  0.5 * np.einsum( 'ijnn -> ij', self.potential.d_Hel )

        return d_nucP

   #####################################################################

    def get_timederiv_mapR( self ):
        #Subroutine to calculate the time-derivative of just the mapping position for each bead

        d_mapR =  np.einsum( 'inm,im->in', self.potential.Hel, self.mapP )

        return d_mapR

   #####################################################################

    def get_timederiv_mapP( self ):
        #Subroutine to calculate the time-derivative of just the mapping momentum for each bead

        d_mapP = -np.einsum( 'inm,im->in', self.potential.Hel, self.mapR )

        return d_mapP

   #####################################################################

    def get_2nd_timederiv_mapR( self, d_mapP ):
        #Subroutine to calculate the second time-derivative of just the mapping positions for each bead
        #This assumes that the nuclei are fixed - used in vv style integrators

        d2_mapR = np.einsum( 'inm,im->in', self.potential.Hel, d_mapP )

        return d2_mapR

   #####################################################################

    def get_PE( self ):
        #Subroutine to calculate potential energy associated with mapping variables and nuclear position

        #Internal ring-polymer modes, 0 if there is only one bead (i.e., LSC-IVR)
        if self.nbds > 1:
            engpe = self.potential.calc_rp_harm_eng( self.nucR, self.beta_p, self.mass )
        else:
            engpe = 0

        #State independent term
        engpe += self.potential.calc_state_indep_eng( self.nucR )

        #Update electronic Hamiltonian matrix
        self.potential.calc_Hel(self.nucR)

        #MMST Term
        engpe += 0.5 * np.sum( np.einsum( 'in,inm,im -> i', self.mapR, self.potential.Hel, self.mapR ) )
        engpe += 0.5 * np.sum( np.einsum( 'in,inm,im -> i', self.mapP, self.potential.Hel, self.mapP ) )

        if (self.potype != 'harm_lin_cpl_symmetrized'):
            engpe += -0.5 * np.sum( np.einsum( 'inn -> i', self.potential.Hel ) )

        return engpe

    #####################################################################

    def get_sampling_theta( self ):
        #Subroutine to calculate the energy used in the MC sampling
        #This is different than the PE used for dynamics
    
        #Theta term
        self.get_theta()
        eng = -np.log( np.abs( self.theta ) ) / self.beta_p

        #Gaussian mapping terms
        eng += np.sum( self.mapR**2 + self.mapP**2 ) / self.beta_p

        return eng

    #####################################################################

    def get_sampling_eng( self ):
        pass

    #####################################################################

    def init_lvc_nucdist_infitemp( self, mass, kvec, amat, nlevel ):
        #get the nuc distribution of coordinates and momenta using Eq(20) in Geva's 2020 paper
        #The one-nuc rho is expressed as tanh(\beta \hbar \omega_i)/\pi \hbar * exp(-tanh(\beta \hbar \omega_i)/\hbar \omega_i) * (c + P^2/2 + 1/2 * omega^2 * R^2 + a_i * R_i))
        #when temperature is 0, tanh(infty) = 1
        
        print()
        print( '#########################################################' )
        print( 'Initializing nuclei equilibrium configuration in state', nlevel, 'using exact quantum distribution' )
        print( '#########################################################' )

        self.nucP = np.zeros([self.nbds, self.nnuc])
        self.nucR = np.zeros([self.nbds, self.nnuc])
        for i in range(self.nnuc):
            self.nucP[:, i] = self.rng.normal( loc = 0.0, scale = np.sqrt( 0.5 * np.sqrt(kvec[i]/mass[i]) ), size = self.nbds )
            self.nucR[:, i] = self.rng.normal( loc = - amat[i, nlevel, nlevel] / ( kvec[i]/mass[i]), scale = np.sqrt( 1 / ( 2*np.sqrt(kvec[i]/mass[i]))), size = self.nbds )

    #####################################################################

    def print_data( self, current_time ):
        #Subroutine to calculate and print-out observables of interest

        fmt_str = '%20.8e'

        ###### CALCULATE OBSERVABLES OF INTEREST #######

        #Calculate potential energy associated with mapping variables and nuclear position
        #This also updates the electronic Hamiltonian matrix
        engpe = self.get_PE()

        #Calculate Nuclear Kinetic Energy
        engke = self.potential.calc_nuc_KE( self.nucP, self.mass )

        #Calculate total energy
        etot = engpe + engke

        #Calculate Q_i array (sized [nbds, nstates])

        Q = self.calc_Q_array()

        #Calculate phi array (sized nbds)

        phi = self.calc_phi_fcn()

        #Calculate electronic-state population using wigner estimator
        wig_pop = self.calc_wigner_estimator()

        #Calculate electronic-state population using semi-classical estimator
        semi_pop = self.calc_semiclass_estimator()

        #Calculate the center of mass of each ring polymer
        nucR_com = self.calc_nucR_com()

        #Updates the value of the W function
        #self.get_W()

        #Updates the value of the Theta function
        self.get_theta()

        ######## PRINT OUT EVERYTHING #######
        output    = np.zeros(6+2*self.nstates+self.nnuc)
        output[0] = current_time
        output[1] = etot
        output[2] = engke
        output[3] = engpe
        output[4] = 0
        output[5] = self.theta
        output[6:6+self.nstates] = wig_pop
        output[6+self.nstates:6+2*self.nstates] = semi_pop
        output[6+2*self.nstates:] = nucR_com
        np.savetxt( self.file_output, output.reshape(1, output.shape[0]), fmt_str )
        self.file_output.flush()

        #columns go as bead_1_nuclei_1 bead_1_nuclei_2 ... bead_1_nuclei_K bead_2_nuclei_1 bead_2_nuclei_2 ...
        np.savetxt( self.file_nucR, np.insert( self.nucR.flatten(), 0, current_time ).reshape(1, self.nucR.size+1), fmt_str )
        np.savetxt( self.file_nucP, np.insert( self.nucP.flatten(), 0, current_time ).reshape(1, self.nucP.size+1), fmt_str )

        #columns go as bead_1_state_1 bead_1_state_2 ... bead_1_state_K bead_2_state_1 bead_2_state_2 ...
        np.savetxt( self.file_mapR, np.insert( self.mapR.flatten(), 0, current_time ).reshape(1, self.mapR.size+1), fmt_str )
        np.savetxt( self.file_mapP, np.insert( self.mapP.flatten(), 0, current_time ).reshape(1, self.mapP.size+1), fmt_str )

        #save Q and phi terms
        np.savetxt( self.file_Q, np.insert( Q.flatten(), 0, current_time).reshape(1, -1),fmt_str )
        np.savetxt( self.file_phi, np.insert( phi.flatten(), 0, current_time).reshape(1, -1),fmt_str )

        self.file_nucR.flush()
        self.file_nucP.flush()
        self.file_mapR.flush()
        self.file_mapP.flush()
        self.file_Q.flush()
        self.file_phi.flush()

    #####################################################################

    def run_Gamma_only_MC( self, Nsteps=1000, Nprint=100,  disp_map=0.1, nm_bool=False, resamp=None, freeze_nuc=False ): 
        #Routine to run Monte-Carlo for mapping variables only
        #The sampling funciton is pure Gamma
        #Random displacements done in real-space for mapping variables
       
        #Nsteps   - number of Monte Carlo steps
        #disp_map - max size of random displacement for mapping variables
        #freeze_nuc - freeze nuclear variables for entire mc run

        #Initialize mapping variables to be zero if None specified
        if (self.mapR is None or self.mapP is None):
            print('automatically initializing all mapping variables to zero')
            self.mapR = np.zeros([self.nbds, self.nstates])
            self.mapP = np.zeros([self.nbds, self.nstates])

        #Open output files
        self.file_output = open( 'output.dat', 'w' )
        self.file_mapR   = open( 'mapR.dat','w' )
        self.file_mapP   = open( 'mapP.dat', 'w' )

        print()
        print( '#########################################################' )
        print( 'Running', self.methodname, 'Gamma MC Routine for', Nsteps, 'Steps' )
        print( '#########################################################' )
        print()

        orig_mapR = np.copy( self.mapR )
        orig_mapP = np.copy( self.mapP )
        
        #Calculate energy for initial configuration
        engold = self.get_sampling_theta()

        numacc = 0 #Counter to keep track of number of accepted MC moves
        for step in range( Nsteps ):
            #Print data starting with initial step
            if( np.mod( step, Nprint ) == 0 ):
                print('Writing data at MC step', step, 'for', self.methodname, 'MC routine')
                self.print_MC_data( step, numacc )
                sys.stdout.flush()

            #check if resampling step
            if( np.mod(step + 1, resamp) == 0 ):
                self.mapR = self.rng.normal(scale=1 / np.sqrt(2), size=self.mapR.shape)
                self.mapP = self.rng.normal(scale=1 / np.sqrt(2), size=self.mapP.shape)
            
            self.mapR += self.rng.uniform(-1.0, 1.0, (self.nbds, self.nstates) ) * disp_map
            self.mapP += self.rng.uniform(-1.0, 1.0, (self.nbds, self.nstates) ) * disp_map

            #Calculate energy of trial position
            engnew = self.get_sampling_theta()
            d_eng = engnew - engold

            #Check acceptance condition
            if d_eng < 0:
                #accept new configuration
                numacc += 1
                orig_mapR = np.copy(self.mapR)
                orig_mapP = np.copy(self.mapP)
                engold    = engnew
            else:
                acc_cond = np.exp( -self.beta_p * ( d_eng ) )
                if( self.rng.random() < acc_cond ):
                    #accept new configuration
                    numacc += 1
                    orig_mapR = np.copy(self.mapR)
                    orig_mapP = np.copy(self.mapP)
                    engold    = engnew
                else:
                    #reject new configuration and reset positions to original position
                    self.mapR  = np.copy( orig_mapR )
                    self.mapP  = np.copy( orig_mapP )

        #Print data at final step regardless of Nprint
        print('Writing data at MC step', step+1, 'for', self.methodname, 'MC routine')
        self.print_MC_data( step+1, numacc )
        sys.stdout.flush()

        #Close output files
        self.file_output.close()
        self.file_mapR.close()
        self.file_mapP.close()

        print()
        print( '#########################################################' )
        print( 'End',self.methodname, 'MC Routine' )
        print( '#########################################################' )
        print()

    #####################################################################

    def print_MC_data( self, step, numacc ):
        #Subroutine to calculate and print-out observables of interest

        fmt_str = '%20.8e'

        ###### CALCULATE OBSERVABLES OF INTEREST #######

        #Calculate sampling gamma terms
        sample_theta = self.get_sampling_theta()

        #Calculate electronic-state population using wigner estimator
        wig_pop = self.calc_wigner_estimator()

        #Calculate electronic-state population using semi-classical estimator
        semi_pop = self.calc_semiclass_estimator()

        #Calculate the center of mass of each ring polymer
        nucR_com = self.calc_nucR_com()

        #Updates the value of the Theta function
        self.get_theta()

        ######## PRINT OUT EVERYTHING #######
        output    = np.zeros(5+2*self.nstates)
        output[0] = step
        output[1] = sample_theta
        output[2] = self.W
        output[3] = self.theta
        if( step == 0 ):
            output[4] = 1.0
        else:
            output[4] = numacc/step
        output[5:5+self.nstates] = wig_pop
        output[5+self.nstates:5+2*self.nstates] = semi_pop
        np.savetxt( self.file_output, output.reshape(1, output.shape[0]), fmt_str )
        self.file_output.flush()

        #columns go as bead_1_state_1 bead_1_state_2 ... bead_1_state_K bead_2_state_1 bead_2_state_2 ...
        np.savetxt( self.file_mapR, np.insert( self.mapR.flatten(), 0, step ).reshape(1, self.mapR.size+1), fmt_str )
        np.savetxt( self.file_mapP, np.insert( self.mapP.flatten(), 0, step ).reshape(1, self.mapP.size+1), fmt_str )

        self.file_mapR.flush()
        self.file_mapP.flush()

    #####################################################################
