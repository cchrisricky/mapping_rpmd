#Define class for nrpmd
#It's a child-class of the map_rpmd parent class

import numpy as np
import utils
import map_rpmd
import sys

class mod_mvrpmd( map_rpmd.map_rpmd ):

    #####################################################################

    def __init__( self, nstates=1, nnuc=1, nbds=1, beta=1.0, mass=1.0, potype=None, potparams=None, mapR=None, mapP=None, nucR=None, nucP=None ):

        super().__init__( 'mod-MV-RPMD', nstates, nnuc, nbds, beta, mass, potype, potparams, mapR, mapP, nucR, nucP )

        #Initalize theta, C, and Hole arrays
        self.theta = None
        self.Cmats = np.zeros( [ nbds, nstates, nstates ], dtype=complex )
        self.holemats = np.zeros( [ nbds, nstates, nstates ], dtype=complex )

    #####################################################################

    def get_timederivs( self ):
        #Subroutine to calculate the time-derivatives of all position/momenta

        #Calculate theta - note this also updates C-matrices
        self.get_theta()

        #Calculate time-derivative of nuclear position
        dnucR = self.get_timederiv_nucR()

        #Calculate hole-matrix for each bead
        self.get_holemats()

        #Calculate nuclear derivative of electronic Hamiltonian matrix
        self.potential.calc_Hel_deriv( self.nucR )

        #Calculate time-derivative of nuclear momenta
        dnucP = self.get_timederiv_nucP()

        #Calculate time-derivative of mapping position and momentum
        dmapR, dmapP = self.get_timederiv_mapRP()

        return dnucR, dnucP, dmapR, dmapP

    #####################################################################

    def get_theta( self ):
        #Subroutine to calculate theta, updates C-matrices

        #Update C-matrices
        self.get_Cmats()

        #Calculate theta
        self.theta = np.real( np.trace( np.linalg.multi_dot( self.Cmats ) ) )

    #####################################################################

    def get_Cmats( self ):
        #Subroutine to calculate C-matrices

        for i in range( self.nbds ):

            vec1 = self.mapR[i,:] + 1j * self.mapP[i,:]
            vec2 = self.mapR[i,:] - 1j * self.mapP[i,:]

            self.Cmats[i,:,:] = np.outer( vec1, vec2 ) - 0.5 * np.eye( self.nstates )

    #####################################################################

    def get_holemats( self ):
        #Subroutine to calculate the Hole matrices following analogy to Bell algorithm in
        #original KC-RPMD paper, Menzeleev, Bell, Miller, JCP 2014
        #Make sure C-matrices have been updated prior to calling this routine

        Fmats = np.zeros([ self.nbds, self.nstates, self.nstates ], dtype=complex)
        Gmats = np.zeros([ self.nbds, self.nstates, self.nstates ], dtype=complex)

        Fmats[0] = np.eye(self.nstates)
        Gmats[-1] = np.eye(self.nstates)

        for i in range(1, self.nbds):
            Fmats[i] = np.dot( Fmats[i - 1], self.Cmats[i - 1] )
        for i in reversed( range(self.nbds - 1) ):
            Gmats[i] = np.dot( self.Cmats[i + 1], Gmats[i + 1] )
        for i in range(self.nbds):
            self.holemats[i] = np.dot( Gmats[i], Fmats[i] )

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

    def get_timederiv_mapRP( self ):
        #Subroutine to calculate the time-derivative of the mapping position and momentum for each bead
        #Note that time derivative is related to derivatives with respect to corresponding mapping momentum and position, respectively

        #Update electronic Hamiltonian matrix
        self.potential.calc_Hel(self.nucR)

        #NRPMD equations of motion
        d_mapR =  np.einsum( 'inm,im->in', self.potential.Hel, self.mapP )
        d_mapP = -np.einsum( 'inm,im->in', self.potential.Hel, self.mapR )

        #Derivative of gaussian term
        d_mapR +=  2.0 / self.beta_p * self.mapP
        d_mapP += -2.0 / self.beta_p * self.mapR

        #Derivative of theta term
        for i in range( self.nbds ):
            for j in range( self.nstates ):

                #Find derivative of C matrix
                horz = np.zeros([self.nstates, self.nstates], dtype=complex)
                vert = np.zeros([self.nstates, self.nstates], dtype=complex)

                horz[j] = self.mapR[i] - 1j * self.mapP[i]
                vert[:,j] = (self.mapR[i] + 1j * self.mapP[i]).T

                d_CR = horz + vert        #derivate of C with respect to position
                d_CP = 1j * (horz - vert) #derivative of C with respect to momentum

                #Calculate derivative of theta
                d_thetaR = np.real( np.trace( np.dot( d_CR, self.holemats[i] ) ) )
                d_thetaP = np.real( np.trace( np.dot( d_CP, self.holemats[i] ) ) )

                #Add force associated with theta portion of potential
                #Noting that the time-derivative of the position (momentum) corresponds to the derivative
                #w/respect to the momentum (position)
                d_mapR[i,j] += -1.0 / ( self.beta_p * self.theta ) * d_thetaP
                d_mapP[i,j] +=  1.0 / ( self.beta_p * self.theta ) * d_thetaR

        return d_mapR, d_mapP

    #####################################################################

    def get_timederiv_mapR( self ):
        #Subroutine to calculate the time-derivative of just the mapping position
        #NOTE: Inefficient to separate out the position and momentum derivative functions
        #XXX: This routine has not been updated because not currently needed for MV-RPMD

        print( 'ERROR: Tried to call get_timederiv_mapR for mod-MV-RPMD, which is currently not coded' )

    #####################################################################

    def get_timederiv_mapP( self ):
        #Subroutine to calculate the time-derivative of just the mapping momentum
        #NOTE: Inefficient to separate out the position and momentum derivative functions
        #XXX: This routine has not been updated because not currently needed for MV-RPMD

        print( 'ERROR: Tried to call get_timederiv_mapP for mod-MV-RPMD, which is currently not coded' )

    #####################################################################

    def get_2nd_timederiv_mapR( self, d_mapP ):
        #Subroutine to calculate the second time-derivative of just the mapping position
        #This assumes that the nuclei are fixed - used in vv style integrators
        #XXX: This routine has not been updated because not currently needed for MV-RPMD

        print( 'ERROR: Tried to call get_2nd_timederiv_mapR for mod-MV-RPMD, which is currently not coded' )

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

        #Gaussian term
        engpe += 1.0/self.beta_p * ( np.sum( self.mapR**2 ) + np.sum( self.mapP**2 ) )

        #Theta term
        self.get_theta()
        engpe -= 1.0/self.beta_p * np.log( np.abs( self.theta ) )

        return engpe

    #####################################################################

    def get_sampling_eng( self ):
        #Subroutine to calculate the energy used in the MC sampling
        #This is the same as the potential energy for MV-RPMD

        return self.get_PE()

    #####################################################################

    def print_data( self, current_time ):
        #Subroutine to calculate and print-out observables of interest

        fmt_str = '%20.8e'

        ###### CALCULATE OBSERVABLES OF INTEREST #######

        #Calculate potential energy associated with mapping variables and nuclear position
        #This also updates theta
        engpe = self.get_PE()

        #Calculate Nuclear Kinetic Energy
        engke = self.potential.calc_nuc_KE( self.nucP, self.mass )

        #Calculate total energy
        etot = engpe + engke

        #Calculate electronic-state population using wigner estimator
        wig_pop = self.calc_wigner_estimator()

        #Calculate electronic-state population using semi-classical estimator
        semi_pop = self.calc_semiclass_estimator()

        #Calculate the center of mass of each ring polymer
        nucR_com = self.calc_nucR_com()

        ######## PRINT OUT EVERYTHING #######
        output    = np.zeros(5+2*self.nstates+self.nnuc)
        output[0] = current_time
        output[1] = etot
        output[2] = engke
        output[3] = engpe
        output[4] = self.theta
        output[5:5+self.nstates] = wig_pop
        output[5+self.nstates:5+2*self.nstates] = semi_pop
        output[5+2*self.nstates:] = nucR_com
        np.savetxt( self.file_output, output.reshape(1, output.shape[0]), fmt_str )
        self.file_output.flush()

        #columns go as bead_1_nuclei_1 bead_1_nuclei_2 ... bead_1_nuclei_K bead_2_nuclei_1 bead_2_nuclei_2 ...
        np.savetxt( self.file_nucR, np.insert( self.nucR.flatten(), 0, current_time ).reshape(1, self.nucR.size+1), fmt_str )
        np.savetxt( self.file_nucP, np.insert( self.nucP.flatten(), 0, current_time ).reshape(1, self.nucP.size+1), fmt_str )

        #columns go as bead_1_state_1 bead_1_state_2 ... bead_1_state_K bead_2_state_1 bead_2_state_2 ...
        np.savetxt( self.file_mapR, np.insert( self.mapR.flatten(), 0, current_time ).reshape(1, self.mapR.size+1), fmt_str )
        np.savetxt( self.file_mapP, np.insert( self.mapP.flatten(), 0, current_time ).reshape(1, self.mapP.size+1), fmt_str )

        self.file_nucR.flush()
        self.file_nucP.flush()
        self.file_mapR.flush()
        self.file_mapP.flush()

    #####################################################################

    def print_MC_data( self, step, numacc ):
        #Subroutine to calculate and print-out observables of interest

        fmt_str = '%20.8e'

        ###### CALCULATE OBSERVABLES OF INTEREST #######

        #Calculate potential energy associated with mapping variables and nuclear position
        #This also updates theta
        engpe = self.get_sampling_eng()

        #Calculate electronic-state population using wigner estimator
        wig_pop = self.calc_wigner_estimator()

        #Calculate electronic-state population using semi-classical estimator
        semi_pop = self.calc_semiclass_estimator()

        #Calculate the center of mass of each ring polymer
        nucR_com = self.calc_nucR_com()

        ######## PRINT OUT EVERYTHING #######
        output    = np.zeros(4+2*self.nstates+self.nnuc)
        output[0] = step
        output[1] = engpe
        output[2] = self.theta
        if( step == 0 ):
            output[3] = 1.0
        else:
            output[3] = numacc/step
        output[4:4+self.nstates] = wig_pop
        output[4+self.nstates:4+2*self.nstates] = semi_pop
        output[4+2*self.nstates:] = nucR_com
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
