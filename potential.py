#subroutine to calculate potential terms

import numpy as np
import utils
from abc import ABC, abstractmethod
import nuc_only_potential

#############################################

def set_potential( potype, potparams, nstates, nnuc, nbds ):

    #Separate routine which returns the appropriate potential class indicated by potype

    potype_list = ['harm_const_cpl', 'harm_lin_cpl', 'harm_lin_cpl_symmetrized' 'nstate_morse', 'nuc_only_harm', 'pengfei_polariton', 'isolated_elec']

    if( potype not in potype_list ):
        print("ERROR: potype not one of valid types:")
        print( *potype_list, sep="\n" )
        exit()

    if( potype[:8] != "nuc_only" ):
        #Initialize a multi-state potential
        potclass = eval( potype + '( potparams, nstates, nnuc, nbds )' )
    else:
        #Initialize a nuclear only potential
        potclass = eval( 'nuc_only_potential.' + potype + '( potparams, nnuc, nbds)' )

    print( 'The potential has been set to',potclass.potname )

    return potclass

######### PARENT POTENTIAL CLASS ##########

class potential(ABC):

    #####################################################################

    @abstractmethod
    def __init__( self, potname, potparams, nstates, nnuc, nbds ):

        self.potname   = potname #string corresponding to the name of the potential
        self.potparams = potparams #array defining the necessary constants for the potential
        self.nstates   = nstates #number of electronic states
        self.nnuc      = nnuc #number of nuclei
        self.nbds      = nbds #number of beads

        #Initialize set of electronic Hamiltonian matrices and they're nuclear derivatives
        self.Hel   = np.zeros( [ nbds, nstates, nstates ] )
        self.d_Hel = np.zeros( [ nbds, nnuc, nstates, nstates ] )

    #####################################################################

    def calc_rp_harm_eng( self, nucR, beta_p, mass ):

        #Calculate potential energy associated with harmonic springs between beads

        engpe = 0.0
        for i in range(self.nbds):
            if( i == 0 ):
                #periodic boundary conditions for the first bead
                engpe += 0.5 * (1.0/beta_p)**2 * np.sum( mass * ( nucR[i] - nucR[self.nbds-1] )**2 )
            else:
                engpe += 0.5 * (1.0/beta_p)**2 * np.sum( mass * ( nucR[i] - nucR[i-1] )**2 )

        return engpe

    ###############################################################

    def calc_rp_harm_force( self, nucR, beta_p, mass ):

        #Calculate force associated with harmonic springs between beads

        Fharm = np.zeros( [self.nbds, self.nnuc] )

        for i in range(self.nbds):
            if( i == 0 ):
                #periodic boundary conditions for the first bead
                Fharm[i] = -mass * (1.0/beta_p)**2 * ( 2.0 * nucR[i] - nucR[self.nbds-1] - nucR[i+1] )
            elif( i == self.nbds-1 ):
                #periodic boundary conditions for the last bead
                Fharm[i] = -mass * (1.0/beta_p)**2 * ( 2.0 * nucR[i] - nucR[i-1] - nucR[0] )
            else:
                Fharm[i] = -mass * (1.0/beta_p)**2 * ( 2.0 * nucR[i] - nucR[i-1] - nucR[i+1] )

        return Fharm

    #####################################################################

    def calc_nuc_KE( self, nucP, mass ):

        #Calculate kinetic energy associated with nuclear beads

        engke = 0.5 * np.sum( nucP**2 / mass )

        return engke

    ###############################################################

    def error_wrong_param_numb( self, num ):

        print("ERROR: List potparams does not have enough entries (",num,") for", self.potname,"potential")
        exit()

    ###############################################################

    @abstractmethod
    def calc_Hel( self ):
        pass

    ###############################################################

    @abstractmethod
    def calc_Hel_deriv( self ):
        pass

    ###############################################################

    @abstractmethod
    def calc_state_indep_eng( self ):
        pass

    ###############################################################

    @abstractmethod
    def calc_state_indep_force( self ):
        pass

    ###############################################################

    @abstractmethod
    def error_check( self ):
        pass

    ###############################################################


####### DEFINED POTENTIALS AS INSTANCES OF PARENT POTENTIAL CLASS #######

class nstate_morse(potential):

    #Class for n-state morse potential with gaussian coupling, see Nandini JPC Lett 2013 and Pengfei JCP 2019

    ###############################################################

    def __init__( self, potparams, nstates, nnuc, nbds ):

        super().__init__( 'n-state morse', potparams, nstates, nnuc, nbds )

        #Set appropriate potential parameters
        if( len(potparams) != 4 ):
            super().error_wrong_param_numb(4)

        self.Dmat     = potparams[0]
        self.alphamat = potparams[1]
        self.Rmat     = potparams[2]
        self.cvec     = potparams[3]

        #Input error check
        self.error_check()

    ###############################################################

    def calc_Hel( self, nucR ):

        #Subroutine to calculate set of electronic Hamiltonian matrices for each bead
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        self.Hel.fill(0.0)

        for i in range( self.nbds ):
            for j in range( self.nnuc ):

                #position of bead i of nuclei j
                pos     = nucR[i,j]

                #nstate x nstate matrix of position minus R-parameter matrix
                posDif  = ( pos - self.Rmat )

                #calculate vector of diagonal terms of Hel
                diag    = np.diag( self.Dmat ) * ( 1.0 - np.exp( - np.diag(self.alphamat) * np.diag(posDif) ) )**2 + self.cvec

                #calculate vector and then convert to upper-triangle matrix of off-diagonal terms of hel
                iup              = np.triu_indices( self.nstates, 1 )
                offdiag          = self.Dmat[iup] * np.exp( - self.alphamat[iup] * posDif[iup]**2 )
                offdiag_mat      = np.zeros([self.nstates,self.nstates])
                offdiag_mat[iup] = offdiag

                #combine diagonal and off diagonal terms into symmetric Hel
                #adding contributions for each nuclei
                self.Hel[i] += np.diag(diag) + offdiag_mat + offdiag_mat.transpose()

    ###############################################################

    def calc_Hel_deriv( self, nucR ):

        #Subroutine to calculate set of nuclear derivative of electronic Hamiltonian matrices for each bead
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        self.d_Hel.fill(0.0)

        for i in range( self.nbds ):
            for j in range( self.nnuc ):

                #position of bead i of nuclei j
                pos     = nucR[i,j]

                #nstate x nstate matrix of position minus R-parameter matrix
                posDif  = ( pos - self.Rmat )

                #calculate vector of diagonal terms of derivative of Hel
                expvec  = np.exp( - np.diag(self.alphamat) * np.diag(posDif) )
                diag    = 2.0* np.diag( self.Dmat ) * np.diag( self.alphamat ) * ( 1.0 - expvec ) * expvec

                #calculate vector and then convert to upper-triangle matrix of off-diagonal terms of derivative of Hel
                iup              = np.triu_indices( self.nstates, 1 )
                offdiag          = -2.0 * self.Dmat[iup] * self.alphamat[iup] * posDif[iup] * np.exp( - self.alphamat[iup] * posDif[iup]**2 )
                offdiag_mat      = np.zeros([self.nstates,self.nstates])
                offdiag_mat[iup] = offdiag

                #combine diagonal and off diagonal terms into symmetric derivative of Hel
                self.d_Hel[i,j] = np.diag(diag) + offdiag_mat + offdiag_mat.transpose()

    ###############################################################

    def calc_state_indep_eng( self, nucR ):
        #Subroutine to calculate the energy associated with the state independent term
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        #There is no state independent term for this potential

        return 0.0

    ###############################################################

    def calc_state_indep_force( self, nucR ):
        #Subroutine to calculate the force associated with the state independent term
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        #There is no state independent term for this potential

        return np.zeros_like(nucR)

    ###############################################################

    def error_check( self ):

        if( self.Dmat.shape != (self.nstates,self.nstates) ):
            print("ERROR: 1st entry of list potparams should correspond to nstate x nstate D-matrix for n-state morse potential")
            exit()

        if( self.alphamat.shape != (self.nstates,self.nstates) ):
            print("ERROR: 2nd entry of list potparams should correspond to nstate x nstate alpha-matrix for n-state morse potential")
            exit()

        if( self.Rmat.shape != (self.nstates,self.nstates) ):
            print("ERROR: 3rd entry of list potparams should correspond to nstate x nstate R-matrix for n-state morse potential")
            exit()

        if( self.cvec.shape != (self.nstates,) ):
            print("ERROR: 4th entry of list potparams should correspond to nstate-dimensional c-vector for n-state morse potential")
            exit()

#########################################################################

class harm_const_cpl(potential):

    #Class for shifted harmonics with a constant coupling between all states
    #Force constant is same between states, but can differ for different nuclei
    #V = \sum_i 0.5 * k_i * R_i**2 + \sum_i H_el(R_i)
    #with [H_el]_nn = a_in * R_i + c_n / nnuc
    #and [H_el]_nm = delta_nm
    #If 2-states it's the usual spin-boson model with constant coupling

    ###############################################################

    def __init__( self, potparams, nstates, nnuc, nbds ):

        super().__init__( 'shifted harmonics - constant coupling', potparams, nstates, nnuc, nbds )

        #Set appropriate potential parameters
        if( len(potparams) != 4 ):
            super().error_wrong_param_numb(4)

        self.kvec = potparams[0] #force constants, size nnuc
        self.avec = potparams[1] #linear-coupling to nuclear modes (shift in harmonic potentials), size nnuc x nstates
        self.cvec = potparams[2] #energy shift for different states, size nstates
        self.deltavec = potparams[3] #vector of electronic couplings, vector which should unpack into upper-triangle of electronic hamiltonian

        #Input error check
        self.error_check()

    ###############################################################

    def calc_Hel( self, nucR ):
        #Subroutine to calculate set of electronic Hamiltonian matrices for each bead
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        self.Hel.fill(0.0)

        for i in range( self.nbds ):

            #Constant electronic coupling
            iup              =  np.triu_indices( self.nstates, 1 )
            self.Hel[i][iup] =  self.deltavec
            self.Hel[i]      += self.Hel[i].T

            for j in range( self.nnuc ):

                #linear coupling to nuclear modes
                self.Hel[i] += np.diag( self.avec[j,:] * nucR[i,j] )

            #Vertical energy shift for states
            self.Hel[i] += np.diag( self.cvec )

    ###############################################################

    def calc_Hel_deriv( self, nucR ):

        #Subroutine to calculate set of nuclear derivative of electronic Hamiltonian matrices for each bead
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        self.d_Hel.fill(0.0)

        #linear coupling to nuclear modes leads to same derivative for all beads for each nuclei
        for j in range( self.nnuc ):
            self.d_Hel[:,j] = np.diag( self.avec[j,:] )

    ###############################################################

    def calc_state_indep_eng( self, nucR ):
        #Subroutine to calculate the energy associated with the state independent term

        #harmonic term with different k for each nuclei
        eng = 0.5 * np.sum( self.kvec * nucR**2 )

        return eng

    ###############################################################

    def calc_state_indep_force( self, nucR ):
        #Subroutine to calculate the force associated with the state independent term

        #force from harmonic term with different k for each nuclei
        force = -self.kvec * nucR

        return force

    ###############################################################

    def error_check( self ):

        if( self.kvec.shape != (self.nnuc,) ):
            print("ERROR: 1st entry of list potparams should correspond to nnuc k-vector for constant coupling harmonic potential")
            exit()

        if( self.avec.shape != (self.nnuc,self.nstates) ):
            print("ERROR: 2nd entry of list potparams should correspond to nnuc x nstate harmonic-shift matrix for constant coupling harmonic potential")
            exit()

        if( self.cvec.shape != (self.nstates,) ):
            print("ERROR: 3rd entry of list potparams should correspond to nstate state energy shift vector for constant coupling harmonic potential")
            exit()

        if( self.deltavec.shape != ( (self.nstates-1)*self.nstates/2,) ):
            print("ERROR: 4th entry of list potparams should correspond to (nstate-1)*nstate/2 coupling vector corresponding to upper triangle of hamiltonian for constant coupling harmonic potential")
            exit()

#########################################################################

class harm_lin_cpl(potential):

    #Class for shifted harmonics where electronic coupling depends linearly on nuclear modes
    #Force constant is same between states, but can differ for different nuclei
    #setting linear couplings to zero reproduces constant coupling potential above
    #See Tamura, Ramon, Bittner, and Burghardt, PRL 2008

    ###############################################################

    def __init__( self, potparams, nstates, nnuc, nbds ):

        super().__init__( 'shifted harmonics - linear coupling', potparams, nstates, nnuc, nbds )

        #Set appropriate potential parameters
        if( len(potparams) != 3 ):
            super().error_wrong_param_numb(3)

        self.kvec = potparams[0] #force constants, size nnuc, NOTE THAT THIS IS FORCE CONSTANT NOT FREQUENCY!!
        self.amat = potparams[1] #shift in harmonic potential along diagonal, and linear couplings along off-diagonal, size nnuc x nstates x nstates (corresponds to kappa and lambda terms in PRL paper)
        #shift in harmonic potentials, size nnuc x nstates (corresponds to kappa terms in PRL paper)
        self.cmat = potparams[2] #energy shift and constant coupling for different states, size nstates x nstates (corresponds to C-matrix in PRL paper)

        #Input error check
        self.error_check()

    ###############################################################

    def calc_Hel( self, nucR ):
        #Subroutine to calculate set of electronic Hamiltonian matrices for each bead
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        self.Hel = np.einsum( 'ijk,ai->ajk', self.amat, nucR )
        self.Hel += self.cmat

    ###############################################################

    def calc_Hel_deriv( self, nucR ):

        #Subroutine to calculate set of nuclear derivative of electronic Hamiltonian matrices for each bead
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        #linear coupling to nuclear modes leads to same derivative for all beads for each nuclei
        self.d_Hel = self.amat[np.newaxis,:,:,:]

    ###############################################################

    def calc_state_indep_eng( self, nucR ):
        #Subroutine to calculate the energy associated with the state independent term

        #harmonic term with different k for each nuclei
        eng = 0.5 * np.sum( self.kvec * nucR**2 )

        return eng

    ###############################################################

    def calc_state_indep_force( self, nucR ):
        #Subroutine to calculate the force associated with the state independent term
        #Note that this corresponds to the negative derivative

        #force from harmonic term with different k for each nuclei
        force = -self.kvec * nucR

        return force

    ###############################################################

    def error_check( self ):

        if( self.kvec.shape != (self.nnuc,) ):
            print("ERROR: 1st entry of list potparams should correspond to nnuc k-vector for linear coupling harmonic potential")
            exit()

        if( self.amat.shape != (self.nnuc,self.nstates,self.nstates) ):
            print("ERROR: 2nd entry of list potparams should correspond to nnuc x nstate x nstate harmonic-shift and linear-coupling tensor for linear coupling harmonic potential")
            exit()

        if( self.cmat.shape != (self.nstates,self.nstates) ):
            print("ERROR: 3rd entry of list potparams should correspond to nstate x nstate state constant energy/coupling matrix for linear coupling harmonic potential")
            exit()

#########################################################################

class harm_lin_cpl_symmetrized(potential):

    #Class for shifted harmonics where electronic coupling depends linearly on nuclear modes
    #Force constant is same between states, but can differ for different nuclei
    #setting linear couplings to zero reproduces constant coupling potential above
    #See Tamura, Ramon, Bittner, and Burghardt, PRL 2008

    ###############################################################

    def __init__( self, potparams, nstates, nnuc, nbds ):

        super().__init__( 'shifted harmonics - linear coupling - sym', potparams, nstates, nnuc, nbds )

        #Set appropriate potential parameters
        if( len(potparams) != 3 ):
            super().error_wrong_param_numb(3)

        self.kvec = potparams[0] #force constants, size nnuc, NOTE THAT THIS IS FORCE CONSTANT NOT FREQUENCY!!
        self.amat = potparams[1] #shift in harmonic potential along diagonal, and linear couplings along off-diagonal, size nnuc x nstates x nstates (corresponds to kappa and lambda terms in PRL paper)
        #shift in harmonic potentials, size nnuc x nstates (corresponds to kappa terms in PRL paper)
        self.cmat = potparams[2] #energy shift and constant coupling for different states, size nstates x nstates (corresponds to C-matrix in PRL paper)

        #Input error check
        self.error_check()

    ###############################################################

    def calc_Hel( self, nucR ):
        #Subroutine to calculate set of electronic Hamiltonian matrices for each bead
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        self.Hel = np.einsum( 'ijk,ai->ajk', self.amat, nucR )
        self.Hel += self.cmat
        
        #minus the average of diagonal terms
        for i in range(self.nbeads):
            self.Hel[i,:,:] -= np.mean(np.diag(self.Hel[i,:,:])) * np.eye(self.nstates)
        
    ###############################################################

    def calc_Hel_deriv( self, nucR ):

        #Subroutine to calculate set of nuclear derivative of electronic Hamiltonian matrices for each bead
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        #linear coupling to nuclear modes leads to same derivative for all beads for each nuclei
        
        #one-bead d_Hel matrix
        d_Hel = np.copy(self.amat)

        for i in range(self.nnuc):
            dHel[i,:,:] -= np.mean(np.diag(self.amat[i,:,:])) * np.eye(self.nstates)
        #multi-bead d_Hel
        self.d_Hel = dHel[np.newaxis,:,:,:]

    ###############################################################

    def calc_state_indep_eng( self, nucR ):
        #Subroutine to calculate the energy associated with the state independent term

        #harmonic term with different k for each nuclei
        eng = 0.5 * np.sum( self.kvec * nucR**2 )

        #V/bar (R): the average or diagonal terms
        eng += np.sum( np.einsum( 'ijj, ai', self.amat, nucR) ) / self.nstates

        return eng

    ###############################################################

    def calc_state_indep_force( self, nucR ):
        #Subroutine to calculate the force associated with the state independent term
        #Note that this corresponds to the negative derivative

        #force from harmonic term with different k for each nuclei
        force = -self.kvec * nucR

        #minus the average-of-diagonal terms
        avg_amat = np.einsum( 'ijj, ai -> ai', self.amat, np.ones_like(nucR) ) / self.nstates
        force -= avg_amat

        return force

    ###############################################################

    def error_check( self ):

        if( self.kvec.shape != (self.nnuc,) ):
            print("ERROR: 1st entry of list potparams should correspond to nnuc k-vector for linear coupling harmonic potential")
            exit()

        if( self.amat.shape != (self.nnuc,self.nstates,self.nstates) ):
            print("ERROR: 2nd entry of list potparams should correspond to nnuc x nstate x nstate harmonic-shift and linear-coupling tensor for linear coupling harmonic potential")
            exit()

        if( self.cmat.shape != (self.nstates,self.nstates) ):
            print("ERROR: 3rd entry of list potparams should correspond to nstate x nstate state constant energy/coupling matrix for linear coupling harmonic potential")
            exit()

#########################################################################

class pengfei_polariton(potential):

    #Class for pengfei's polariton model
    #A. Mandal, P. Huo JPC Lett 2019
    #Just work with Hamiltonian restricted to |e,0> and |g,1> subspace

    ###############################################################

    def __init__( self, potparams, nstates, nnuc, nbds ):

        super().__init__( 'pengfei_polariton', potparams, nstates, nnuc, nbds )

        #Set appropriate potential parameters
        if( len(potparams) != 6 ):
            super().error_wrong_param_numb(6)

        self.wc = potparams[0] #frequency of photon mode
        self.gc = potparams[1] #coupling to photon mode
        self.Amat = potparams[2] #A-terms in electronic states, first index is the adiabatic state ie A[0,0]=A1, A[0,1]=A2, A[1,0]=A3, A[1,1]=A4 in paper
        self.Bmat = potparams[3] #B-terms as above
        self.Rmat = potparams[4] #R-terms as above
        self.Dvec = potparams[5] #D-terms with D[0]=D1 and D[1]=D2 from paper

        #Input error check
        self.error_check()

    ###############################################################

    def calc_Hel( self, nucR ):
        #Subroutine to calculate set of electronic Hamiltonian matrices for each bead
        #Assumes first nuclei corresponds to reaction coordinate
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        #Calculate the adiabatic electronic energy for each bead
        #Eg = self.calc_adiabatic_energy( nucR[:,0], 0)
        #Ee = self.calc_adiabatic_energy( nucR[:,0], 1)

        rxnR = np.copy( nucR[:,0] )

        Vmat = np.zeros([self.nbds,self.nstates,self.nstates])

        Vmat[:,0,0] = self.Amat[0,0] + self.Bmat[0,0] * ( rxnR - self.Rmat[0,0] )**2
        Vmat[:,0,1] = self.Amat[0,1] + self.Bmat[0,1] * ( rxnR - self.Rmat[0,1] )**2
        Vmat[:,1,0] = self.Amat[1,0] + self.Bmat[1,0] * ( rxnR - self.Rmat[1,0] )**2
        Vmat[:,1,1] = self.Amat[1,1] + self.Bmat[1,1] * ( rxnR - self.Rmat[1,1] )**2

        self.Hel = np.zeros([self.nbds,self.nstates,self.nstates])

        self.Hel[:,0,0] = 0.5*( Vmat[:,0,0] + Vmat[:,0,1] ) - np.sqrt( self.Dvec[0]**2 + 0.25* ( Vmat[:,0,0] - Vmat[:,0,1] )**2 )
        self.Hel[:,1,1] = 0.5*( Vmat[:,1,0] + Vmat[:,1,1] ) - np.sqrt( self.Dvec[1]**2 + 0.25* ( Vmat[:,1,0] - Vmat[:,1,1] )**2 )

        #Need to add all the gc wc stuff to the hamiltonian

        exit()


    ###############################################################

    def calc_Hel_deriv( self, nucR ):

        #Subroutine to calculate set of nuclear derivative of electronic Hamiltonian matrices for each bead
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        self.d_Hel = 2.0

    ################################################################

    def calc_state_indep_eng( self, nucR ):
        #Subroutine to calculate the energy associated with the state independent term

        eng = 1.0

        return eng

    ################################################################

    def calc_state_indep_force( self, nucR ):
        #Subroutine to calculate the force associated with the state independent term
        #Note that this corresponds to the negative derivative

        #force from harmonic term with different k for each nuclei
        force = -1.0

        return force

    ###############################################################

    def error_check( self ):

        if( self.nstates != 2 ):
            print("ERROR: pengfei_polariton only supports two electronic states")

        if( not isinstance(self.wc, float) ):
            print("ERROR: 1st entry of list potparams should be the frequency of photon mode")
            exit()

        if( not isinstance(self.gc, float) ):
            print("ERROR: 2nd entry of list potparams should be the constant coupling to photon mode")
            exit()

        if( self.Amat.shape != (2,2) ):
            print("ERROR: 3rd entry of list potparams should correspond to 2x2 array of A-terms in electronic states")
            exit()

        if( self.Bmat.shape != (2,2) ):
            print("ERROR: 4th entry of list potparams should correspond to 2x2 array of B-terms in electronic states")
            exit()

        if( self.Rmat.shape != (2,2) ):
            print("ERROR: 5th entry of list potparams should correspond to 2x2 array of R-terms in electronic states")
            exit()

        if( self.Dvec.shape != (2,) ):
            print("ERROR: 6th entry of list potparams should correspond to 2-dim vector of D-terms in electronic states")
            exit()

#########################################################################

class isolated_elec(potential):

    #class for isolated constant electronic potential
    #electronic hamiltonian is constant wrt to all variables
    #no state independent potential either

    def __init__(self, potparams, nstates, nnuc, nbds):

        super().__init__( 'isolated electronic', potparams, nstates, nnuc, nbds )

        #Set appropriate potential parameters
        if( len(potparams) != 1 ):
            super().error_wrong_param_numb(1)

        self.Helec = potparams[0] #electronic hamiltonian that is constant

        #Input error check
        self.error_check()

        self.Hel[:] = self.Helec #set all bead hamiltonians to be the same
        self.d_Hel.fill(0.0) #since hamiltonian is constant the derivative is 0

    def calc_Hel(self, *args):
        #hamiltonian is constant so no need to update it
        pass

    def calc_Hel_deriv(self, *args):
        #hamiltonian derivative is constant so no need to update it
        pass

    def calc_state_indep_eng(self, *args):
        #no state independent potential
        return 0.0

    def calc_state_indep_force(self, *args):
        #no state independent potential
        return 0.0

    def error_check(self):
        if (self.Helec.shape != (self.nstates, self.nstates)):
            print("ERROR: 1st entry of list potparams should correspond to full constant electronic hamiltonian")
            exit()
