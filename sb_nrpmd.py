#Define class for nrpmd
#It's a child-class of the map_rpmd parent class
#The class of single bead nrpmd dynamics
#Here the nuclear DOFs are treated by path-integral, with nbds copies of RP.
#The electronic mapping variables only has one bead
#The vibronic coupling only happens for the last nuclear bead

import numpy as np
import utils
import map_rpmd
import sys

class sb_nrpmd( map_rpmd.map_rpmd ):

    #####################################################################

    def __init__( self, nstates=1, nnuc=1, nbds=1, beta=1.0, mass=1.0, potype=None, potparams=None, mapR=None, mapP=None, nucR=None, nucP=None, Heltype=None ):
        
        self.beta_p = beta / nbds
        self.Heltype = Heltype
        super().__init__( 'sb-NRPMD', nstates, nnuc, nbds, beta, mass, potype, potparams, mapR, mapP, nucR, nucP )

        self.init_error_check()
    #####################################################################

    def init_error_check( self ):

        #Input Error Checks
        if( self.nucR is not None and self.nucR.shape != (self.nbds, self.nnuc) ):
            print('ERROR: Size of nuclear position array doesnt match bead number or number of nuclei')
            exit()

        if( self.nucP is not None and self.nucP.shape != (self.nbds, self.nnuc) ):
            print('ERROR: Size of nuclear momentum array doesnt match bead number or number of nuclei')
            exit()

        if( self.mapR is not None and self.mapR.shape != (self.nstates,) ):
            print('ERROR: Size of mapping position array doesnt match number of states')
            exit()

        if( self.mapP is not None and self.mapP.shape != (self.nstates,) ):
            print('ERROR: Size of mapping momentum array doesnt match number of states')
            exit()

        if( self.mass.shape[0] != self.nnuc ):
            print('ERROR: Size of nuclear mass array doesnt match number of nuclei')
            exit()

        if( self.Heltype not in ['ave', 'last'] ):
            print('ERROR: invalid Hel type')
            exit()

    #####################################################################

    def get_nucP_MB( self, beta=None ):

        #Obtain nuclear momentum from Maxwell-Boltzmann distribution at beta
        #distribution defined as e^(-1/2*x^2/sigma^2) so sigma=sqrt(mass/beta_p)

        if (beta is None):
            beta_p = self.beta_p
        else:
            beta_p = beta / self.nbds
        self.nucP = np.zeros([self.nbds,self.nnuc])

        for i in range( self.nnuc ):
            sigma = np.sqrt( self.mass[i] / beta_p )
            self.nucP[:,i] = self.rng.normal( 0.0, sigma, self.nbds )

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

    def get_map_phi(self):

        #Initialize mapping variables by pulled from phi distribution
        #phi = 2^(L+2) exp[-\sum_i (x_i^2 + p_i^2)]
        #No state specificity
        #See Geva JCTC 2020

        print()
        print( '#########################################################' )
        print( 'Initializing Mapping Variables using Wigner sampling' )
        print( '#########################################################' )
        print()

        self.mapR = self.rng.normal( 0.0, np.sqrt(0.5), self.nstates )
        self.mapP = self.rng.normal( 0.0, np.sqrt(0.5), self.nstates )

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

    def calc_wigner_estimator( self ):

        #Calculate poulation of each electronic state using the Wigner Estimator
        #See Duke and Ananth JPC Lett 2015

        fctr = 2.0**(self.nstates+1) * np.exp( - np.sum( self.mapR**2 + self.mapP**2 ) )
        pop  = fctr * ( self.mapR**2 + self.mapP**2 - 0.5 )

        return pop

    #####################################################################

    def calc_semiclass_estimator( self ):

        #Calculate poulation of each electronic state using the semi-classical Estimator
        #See Chowdhury and Huo JCP 2019

        pop = np.sum( 0.5*( self.mapR**2 + self.mapP**2 - 1.0 ), 0 )

        return pop

    #####################################################################

    def get_gamma_gauss( self ):
        #Calculate the gaussian form of the gamma partition function prefactor
        #equivalent to a sum over the wigner population estimator

        fctr  = 2.0**(self.nstates+1) * np.exp( - np.sum( self.mapR**2 + self.mapP**2 ) )
        gamma = fctr * ( np.sum(self.mapR**2 + self.mapP**2) - 0.5*self.nstates )

        return gamma

    #####################################################################

    def get_gamma_semi( self ):
        #Calculate the semiclassical form of the gamma partition function prefactor
        #equivalent to a sum over the semiclassical population estimator

        gamma = 0.5 * ( np.sum(self.mapR**2 + self.mapP**2) - self.nstates )

        return gamma

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
        #only the last bead is influenced by state dependent Hamiltonian
        if self.Heltype == 'last':
            d_nucP[-1] += -0.5 * np.einsum( 'n,inm,m -> i', self.mapR, self.potential.d_Hel[-1], self.mapR )
            d_nucP[-1] += -0.5 * np.einsum( 'n,inm,m -> i', self.mapP, self.potential.d_Hel[-1], self.mapP )
            d_nucP[-1] +=  0.5 * np.einsum( 'inn -> i', self.potential.d_Hel[-1] )
        if self.Heltype == 'ave':
            d_nucP += -0.5 * np.einsum( 'n,ijnm,m -> ij', self.mapR, self.potential.d_Hel, self.mapR ) / self.nbds
            d_nucP += -0.5 * np.einsum( 'n,ijnm,m -> ij', self.mapP, self.potential.d_Hel, self.mapP ) / self.nbds
            d_nucP +=  0.5 * np.einsum( 'ijnn -> ij', self.potential.d_Hel ) / self.nbds
        return d_nucP

   #####################################################################

    def get_timederiv_mapR( self ):
        #Subroutine to calculate the time-derivative of just the mapping position for each bead

        if self.Heltype == 'last':
            d_mapR =  np.einsum( 'nm,m->n', self.potential.Hel[-1], self.mapP )
        if self.Heltype == 'ave':
            d_mapR =  np.einsum( 'nm,m->n', np.mean(self.potential.Hel, axis = 0), self.mapP )
        return d_mapR

   #####################################################################

    def get_timederiv_mapP( self ):
        #Subroutine to calculate the time-derivative of just the mapping momentum for each bead

        if self.Heltype == 'last':
            d_mapP = -np.einsum( 'nm,m->n', self.potential.Hel[-1], self.mapR )

        if self.Heltype == 'ave':
            d_mapP = -np.einsum( 'nm,m->n', np.mean(self.potential.Hel, axis = 0), self.mapR )

        return d_mapP

   #####################################################################

    def get_2nd_timederiv_mapR( self, d_mapP ):
        #Subroutine to calculate the second time-derivative of just the mapping positions for each bead
        #This assumes that the nuclei are fixed - used in vv style integrators
        if self.Heltype == 'last':
            d2_mapR =  np.einsum( 'nm,m->n', self.potential.Hel[-1], d_mapP )
        if self.Heltype == 'ave':
            d2_mapR =  np.einsum( 'nm,m->n', np.mean(self.potential.Hel, axis = 0), d_mapP )
        return d2_mapR

    #####################################################################

    def get_sampling_eng( self ):
        pass

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
        if self.Heltype == 'last':
            engpe += 0.5 * np.einsum( 'n,nm,m', self.mapR, self.potential.Hel[-1], self.mapR ) 
            engpe += 0.5 * np.einsum( 'n,nm,m', self.mapP, self.potential.Hel[-1], self.mapP )
            engpe += -0.5 * np.einsum( 'nn', self.potential.Hel[-1] ) 
        if self.Heltype == 'ave':
            engpe += 0.5 * np.einsum( 'n,nm,m', self.mapR, np.mean(self.potential.Hel, axis=0), self.mapR ) 
            engpe += 0.5 * np.einsum( 'n,nm,m', self.mapP, np.mean(self.potential.Hel, axis=0), self.mapP )
            engpe += -0.5 * np.einsum( 'nn', np.mean(self.potential.Hel, axis=0) ) 
        return engpe

    #####################################################################

    def get_theta( self ):
        #Subroutine to calculate Theta (minus the Gaussian term and prefactor) following Huo's 2019 paper https://dx.doi.org/10.1063/1.5096276

        #Electronic trace over the product of C matrices
        halfeye = 0.5 * np.eye(self.nstates)
        prod = np.outer(( self.mapR + 1j*self.mapP ), ( self.mapR - 1j*self.mapP )) - halfeye

        self.theta = np.real( np.trace( prod ) )

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

        #calculate the theta term
        theta = self.get_theta()

        #Calculate electronic-state population using wigner estimator
        wig_pop = self.calc_wigner_estimator()

        #Calculate electronic-state population using semi-classical estimator
        semi_pop = self.calc_semiclass_estimator()

        #Calculate the center of mass of each ring polymer
        nucR_com = self.calc_nucR_com()

        #Calculate Q_i array (sized [nbds, nstates])
        Q = self.calc_Q_array_sb()

        #Calculate phi array (sized nbds)
        phi = self.calc_phi_fcn_sb()

        ######## PRINT OUT EVERYTHING #######
        output    = np.zeros(5+2*self.nstates+self.nnuc)
        output[0] = current_time
        output[1] = etot
        output[2] = engke
        output[3] = engpe
        output[4] = theta
        output[5:5+self.nstates] = wig_pop
        output[5+self.nstates:5+2*self.nstates] = semi_pop
        output[5+2*self.nstates:] = nucR_com
        np.savetxt( self.file_output, output.reshape(1, output.shape[0]), fmt_str )
        self.file_output.flush()

        #columns go as bead_1_nuclei_1 bead_1_nuclei_2 ... bead_1_nuclei_K bead_2_nuclei_1 bead_2_nuclei_2 ...
        np.savetxt( self.file_nucR, np.insert( self.nucR.flatten(), 0, current_time ).reshape(1, self.nucR.size+1), fmt_str )
        np.savetxt( self.file_nucP, np.insert( self.nucP.flatten(), 0, current_time ).reshape(1, self.nucP.size+1), fmt_str )

        #columns go as state_1 state_2 ... state_K
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

    '''def print_MC_data( self, step, numacc ):
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

        #Calulate the value of the gaussian gamma function
        gamma_gauss = np.sum( wig_pop )

        #Calulate the value of the semiclassical gamma function
        gamma_semi = np.sum( semi_pop )

        ######## PRINT OUT EVERYTHING #######
        output    = np.zeros(5+2*self.nstates+self.nnuc)
        output[0] = step
        output[1] = engpe
        if self.gammaType == 'gauss':
            output[2] = gamma_gauss
            output[3] = gamma_semi
        else:
            output[2] = gamma_semi
            output[3] = gamma_gauss
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

        #columns go as state_1 state_2 ... state_K
        np.savetxt( self.file_mapR, np.insert( self.mapR.flatten(), 0, step ).reshape(1, self.mapR.size+1), fmt_str )
        np.savetxt( self.file_mapP, np.insert( self.mapP.flatten(), 0, step ).reshape(1, self.mapP.size+1), fmt_str )

        self.file_nucR.flush()
        self.file_mapR.flush()
        self.file_mapP.flush()

    #####################################################################'''
