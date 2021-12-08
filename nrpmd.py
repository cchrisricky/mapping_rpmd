#Define class for nrpmd
#It's a child-class of the map_rpmd parent class

import numpy as np
import utils
import map_rpmd
import sys
import time
from scipy.linalg import expm

class nrpmd( map_rpmd.map_rpmd ):

    #####################################################################

    def __init__( self, nstates=1, nnuc=1, nbds=1, beta=1.0, mass=1.0, potype=None, potparams=None, mapR=None, mapP=None, nucR=None, nucP=None ):

        super().__init__( 'NRPMD', nstates, nnuc, nbds, beta, mass, potype, potparams, mapR, mapP, nucR, nucP )

        #initilaize W and Theta
        self.W = None
        self.theta = None

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

    def get_theta( self ):
        #Subroutine to calculate Theta (minus the Gaussian term and prefactor) following Huo's 2019 paper https://dx.doi.org/10.1063/1.5096276

        #Electronic trace over the product of C matrices
        halfeye = 0.5 * np.eye(self.nstates)
        prod = np.outer(( self.mapR[0] + 1j*self.mapP[0] ), ( self.mapR[0] - 1j*self.mapP[0] )) - halfeye
        for i in range(1, self.nbds):
            prod = prod @ ( np.outer(( self.mapR[i] + 1j*self.mapP[i] ), ( self.mapR[i] - 1j*self.mapP[i] )) - halfeye )
        self.theta = np.real( np.trace( prod ) )

    #####################################################################

    def get_timederiv_nucR( self ):
        #Subroutine to calculate the time-derivative of the nuclear positions for each bead

        return self.nucP / self.mass

    #####################################################################

    def get_timederiv_nucP( self, intRP_bool=True ):
        #Subroutine to calculate the time-derivative of the nuclear momenta for each bead

        #Force associated with harmonic springs between beads and the state-independent portion of the potential
        #This is dealt with in the parent class
        #If intRP_bool is False does not calculate the contribution from the harmonic ring polymer springs
        d_nucP = super().get_timederiv_nucP( intRP_bool)

        #Calculate nuclear derivative of electronic Hamiltonian matrix
        self.potential.calc_Hel_deriv( self.nucR )

        #Calculate contribution from MMST term
        #XXX could maybe make this faster getting rid of double index in einsum
        d_nucP += -0.5 * np.einsum( 'in,ijnm,im -> ij', self.mapR, self.potential.d_Hel, self.mapR )
        d_nucP += -0.5 * np.einsum( 'in,ijnm,im -> ij', self.mapP, self.potential.d_Hel, self.mapP )
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

        d2_mapR =  np.einsum( 'inm,im->in', self.potential.Hel, d_mapP )

        return d2_mapR

   #####################################################################

    def get_PE( self ):
        #Subroutine to calculate potential energy associated with mapping variables and nuclear position

        #Internal ring-polymer modes
        engpe = self.potential.calc_rp_harm_eng( self.nucR, self.beta_p, self.mass )

        #State independent term
        engpe += self.potential.calc_state_indep_eng( self.nucR )

        #Update electronic Hamiltonian matrix
        self.potential.calc_Hel(self.nucR)

        #MMST Term
        engpe += 0.5 * np.sum( np.einsum( 'in,inm,im -> i', self.mapR, self.potential.Hel, self.mapR ) )
        engpe += 0.5 * np.sum( np.einsum( 'in,inm,im -> i', self.mapP, self.potential.Hel, self.mapP ) )
        engpe += -0.5 * np.sum( np.einsum( 'inn -> i', self.potential.Hel ) )

        return engpe

    #####################################################################

    def get_sampling_eng( self ):
        #Subroutine to calculate the energy used in the MC sampling
        #This is different than the PE used for dynamics

        #W term
        self.get_W()
        eng = -np.log( np.abs( self.W ) ) / self.beta_p

        #Gaussian mapping terms
        eng += np.sum( self.mapR**2 + self.mapP**2 ) / self.beta_p

        #Internal ring-polymer modes
        eng += self.potential.calc_rp_harm_eng( self.nucR, self.beta_p, self.mass )

        #State independent term
        eng += self.potential.calc_state_indep_eng( self.nucR )

        return eng

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
        self.get_W()

        #Updates the value of the Theta function
        self.get_theta()

        ######## PRINT OUT EVERYTHING #######
        output    = np.zeros(6+2*self.nstates+self.nnuc)
        output[0] = current_time
        output[1] = etot
        output[2] = engke
        output[3] = engpe
        output[4] = self.W
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

    def print_MC_data( self, step, numacc ):
        #Subroutine to calculate and print-out observables of interest

        fmt_str = '%20.8e'

        ###### CALCULATE OBSERVABLES OF INTEREST #######

        #Calculate sampling energy (updates W)
        engpe = self.get_sampling_eng()

        #Calculate electronic-state population using wigner estimator
        wig_pop = self.calc_wigner_estimator()

        #Calculate electronic-state population using semi-classical estimator
        semi_pop = self.calc_semiclass_estimator()

        #Calculate the center of mass of each ring polymer
        nucR_com = self.calc_nucR_com()

        #Updates the value of the Theta function
        self.get_theta()

        ######## PRINT OUT EVERYTHING #######
        output    = np.zeros(5+2*self.nstates+self.nnuc)
        output[0] = step
        output[1] = engpe
        output[2] = self.W
        output[3] = self.theta
        if( step == 0 ):
            output[4] = 1.0
        else:
            output[4] = numacc/step
        output[5:5+self.nstates] = wig_pop
        output[5+self.nstates:5+2*self.nstates] = semi_pop
        output[5+2*self.nstates:] = nucR_com
        np.savetxt( self.file_output, output.reshape(1, output.shape[0]), fmt_str )
        self.file_output.flush()

        #columns go as bead_1_nuclei_1 bead_1_nuclei_2 ... bead_1_nuclei_K bead_2_nuclei_1 bead_2_nuclei_2 ...
        np.savetxt( self.file_nucR, np.insert( self.nucR.flatten(), 0, step ).reshape(1, self.nucR.size+1), fmt_str )

        #columns go as bead_1_state_1 bead_1_state_2 ... bead_1_state_K bead_2_state_1 bead_2_state_2 ...
        np.savetxt( self.file_mapR, np.insert( self.mapR.flatten(), 0, step ).reshape(1, self.mapR.size+1), fmt_str )
        np.savetxt( self.file_mapP, np.insert( self.mapP.flatten(), 0, step ).reshape(1, self.mapP.size+1), fmt_str )

        self.file_nucR.flush()
        self.file_mapR.flush()
        self.file_mapP.flush()

    #####################################################################
