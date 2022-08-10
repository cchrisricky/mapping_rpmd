import numpy as np
import normal_mode

class integrator():

    ###############################################################

    def __init__( self, map_rpmd, delt, intype, small_dt_ratio ):

        #map_rpmd is the mapping-rpmd class passed to the routines
        self.delt    = delt #time-step
        self.intype  = intype #integrator type
        self.small_dt_ratio = small_dt_ratio #number of steps taken to update mapping variables for given nuclear update for vv-like integrators

        #Input error check
        self.error_check( map_rpmd )

        print('Using integrator',intype,'with a time-step = ',delt)

        #Want to actually save time-derivative of nuclear momentum from step to step if doing
        #velocity-verlet, analytical, or cayley integration
        if( self.intype == 'vv' or self.intype == 'analyt' or self.intype == 'cayley' ):
            self.d_nucP_for_vv = None
            print( 'Number of steps for internal mapping-variable velocity-verlet loop is = ',small_dt_ratio )


        #Initialize normal mode frequencies of ring polymer if doing analytical or cayley integration
        #Note that normal-mode frequencies are mass independent and thus are the same for all nuclei
        #Thus all normal-mode arrays are of dimension nbds
        if( self.intype == 'analyt' or self.intype == 'cayley' ):

            self.nm_freq = normal_mode.calc_normal_mode_freq( map_rpmd.beta_p, map_rpmd.nbds )

            #if using cayley integrator make additional frequency arrays
            if( self.intype == 'cayley' ):
                self.nm_freq_prod = self.delt * self.nm_freq**2
                self.nm_freq_sum  = 1 + 0.25 * self.delt * self.nm_freq_prod
                self.nm_freq_dif  = 1 - 0.25 * self.delt * self.nm_freq_prod
        
        #Want to store the last 4 derivative if using the ABM integrator
        if( self.intype == 'abm' ):
            self.prev_d_nucR = np.zeros( (4,) + map_rpmd.nucR.shape )
            self.prev_d_nucP = np.zeros( (4,) + map_rpmd.nucP.shape )
            self.prev_d_mapR = np.zeros( (4,) + map_rpmd.mapR.shape )
            self.prev_d_mapP = np.zeros( (4,) + map_rpmd.mapP.shape )

    ###############################################################

    def onestep( self, map_rpmd, step ):

        #Wrapper subroutine to choose appropriate integrator to integrate
        #equations of motion by one time-step

        if( self.intype == 'rk4' ):
            self.rk4( map_rpmd )
        elif( self.intype == 'abm' ):
            self.abm( map_rpmd, step )
        elif( self.intype == 'vv' or self.intype == 'analyt' or self.intype == 'cayley' ):
            self.vv_outer( map_rpmd, step )

    ###############################################################

    def rk4( self, map_rpmd ):

        #Integrate EOM using 4th order runge-kutta
        #Note that step isn't used in this routine

        #Copy initial positions and momenta at time t
        init_nucR = np.copy( map_rpmd.nucR )
        init_nucP = np.copy( map_rpmd.nucP )
        init_mapR = np.copy( map_rpmd.mapR )
        init_mapP = np.copy( map_rpmd.mapP )

        #obtain k1 vectors
        k1_nucR, k1_nucP, k1_mapR, k1_mapP = map_rpmd.get_timederivs()

        #obtain k2 vectors
        map_rpmd.nucR = init_nucR + 0.5 * self.delt * k1_nucR
        map_rpmd.nucP = init_nucP + 0.5 * self.delt * k1_nucP
        map_rpmd.mapR = init_mapR + 0.5 * self.delt * k1_mapR
        map_rpmd.mapP = init_mapP + 0.5 * self.delt * k1_mapP

        k2_nucR, k2_nucP, k2_mapR, k2_mapP = map_rpmd.get_timederivs()

        #obtain k3 vectors
        map_rpmd.nucR = init_nucR + 0.5 * self.delt * k2_nucR
        map_rpmd.nucP = init_nucP + 0.5 * self.delt * k2_nucP
        map_rpmd.mapR = init_mapR + 0.5 * self.delt * k2_mapR
        map_rpmd.mapP = init_mapP + 0.5 * self.delt * k2_mapP

        k3_nucR, k3_nucP, k3_mapR, k3_mapP = map_rpmd.get_timederivs()

        #obtain k4 vectors
        map_rpmd.nucR = init_nucR + 1.0 * self.delt * k3_nucR
        map_rpmd.nucP = init_nucP + 1.0 * self.delt * k3_nucP
        map_rpmd.mapR = init_mapR + 1.0 * self.delt * k3_mapR
        map_rpmd.mapP = init_mapP + 1.0 * self.delt * k3_mapP

        k4_nucR, k4_nucP, k4_mapR, k4_mapP = map_rpmd.get_timederivs()

        #update new positions/momenta using RK4
        map_rpmd.nucR = init_nucR + 1.0/6.0 * self.delt * ( k1_nucR + 2.0*k2_nucR + 2.0*k3_nucR + k4_nucR )
        map_rpmd.nucP = init_nucP + 1.0/6.0 * self.delt * ( k1_nucP + 2.0*k2_nucP + 2.0*k3_nucP + k4_nucP )
        map_rpmd.mapR = init_mapR + 1.0/6.0 * self.delt * ( k1_mapR + 2.0*k2_mapR + 2.0*k3_mapR + k4_mapR )
        map_rpmd.mapP = init_mapP + 1.0/6.0 * self.delt * ( k1_mapP + 2.0*k2_mapP + 2.0*k3_mapP + k4_mapP )

    ###############################################################

    def vv_outer( self, map_rpmd, step ):

        #Outer loop to integrate EOM using velocity-verlet like algorithms
        #This includes pengfei's implementation, and the analtyical and cayley modification of it

        #If initial step of dynamics need to initialize electronic hamiltonian
        #and derivative of nuclear momentum (aka the force on the nuclei)
        #NOTE: moving forward these calls may need to be generalized to allow for other types of methods
        if( step == 0 ):
            map_rpmd.potential.calc_Hel( map_rpmd.nucR )
            if( self.intype == 'vv' ):
                self.d_nucP_for_vv = map_rpmd.get_timederiv_nucP(intRP_bool=True)
            else:
                self.d_nucP_for_vv = map_rpmd.get_timederiv_nucP(intRP_bool=False)

        #Update nuclear momentum by 1/2 a time-step
        self.update_vv_nucP( map_rpmd )

        #Update mapping variables by 1/2 a time-step
        self.update_vv_mapRP( map_rpmd )

        #Update nuclear position for full time-step
        if( self.intype == 'vv' ):
            self.update_vv_nucR( map_rpmd )
        elif( self.intype == 'analyt' ):
            self.update_analyt_nucR( map_rpmd )
        elif( self.intype == 'cayley' ):
            self.update_cayley_nucR( map_rpmd )

        #Update electronic Hamiltonian given new positioninteg
        #Update mapping variables to full time-step
        self.update_vv_mapRP( map_rpmd )

        #Calculate derivative of nuclear momentum at new time-step (aka the force on the nuclei)
        #Don't include contribution from internal modes of ring-polymer if doing analyt or cayley
        if( self.intype == 'vv' ):
            self.d_nucP_for_vv = map_rpmd.get_timederiv_nucP(intRP_bool=True)
        else:
            self.d_nucP_for_vv = map_rpmd.get_timederiv_nucP(intRP_bool=False)

        #Update nuclear momentum to full time-step
        self.update_vv_nucP( map_rpmd )

    def vv_outer_nuconly( self, map_rpmd, step):

        #Outer loop to integrate EOM using velocity-verlet like algorithms
        #This includes pengfei's implementation, and the analtyical and cayley modification of it

        #If initial step of dynamics need to initialize electronic hamiltonian
        #and derivative of nuclear momentum (aka the force on the nuclei)
        #NOTE: moving forward these calls may need to be generalized to allow for other types of methods
        if( step == 0 ):
            map_rpmd.potential.calc_Hel( map_rpmd.nucR )
            if( self.intype == 'vv' ):
                self.d_nucP_for_vv = map_rpmd.get_timederiv_nucP(intRP_bool=True)
            else:
                self.d_nucP_for_vv = map_rpmd.get_timederiv_nucP(intRP_bool=False)

        #Update nuclear momentum by 1/2 a time-step
        self.update_vv_nucP( map_rpmd )

        #Update mapping variables by 1/2 a time-step
        self.update_vv_mapRP( map_rpmd )

        #Update nuclear position for full time-step
        if( self.intype == 'vv' ):
            self.update_vv_nucR( map_rpmd )
        elif( self.intype == 'analyt' ):
            self.update_analyt_nucR( map_rpmd )
        elif( self.intype == 'cayley' ):
            self.update_cayley_nucR( map_rpmd )

        #Calculate derivative of nuclear momentum at new time-step (aka the force on the nuclei)
        #Don't include contribution from internal modes of ring-polymer if doing analyt or cayley
        if( self.intype == 'vv' ):
            self.d_nucP_for_vv = map_rpmd.get_timederiv_nucP(intRP_bool=True)
        else:
            self.d_nucP_for_vv = map_rpmd.get_timederiv_nucP(intRP_bool=False)

        #Update nuclear momentum to full time-step
        self.update_vv_nucP( map_rpmd )

    ###############################################################

    def update_vv_nucP( self, map_rpmd ):

        #Update nuclear momentum by 1/2 a time-step
        map_rpmd.nucP += 0.5 * self.delt * self.d_nucP_for_vv

    ###############################################################

    def update_vv_nucR( self, map_rpmd ):

        #Update nuclear position by a full time-step
        map_rpmd.nucR += map_rpmd.nucP * self.delt / map_rpmd.mass

    ###############################################################

    def update_analyt_nucR( self, map_rpmd ):

        #Update nuclear position by a full time-step using analytical
        #result for internal modes of ring-polymer

        #Transform position and velocities to normal-modes using fourier-transform
        #Note this is faster than directly diagonalizing the frequency matrix
        nucR_nm = np.zeros([map_rpmd.nbds, map_rpmd.nnuc])
        nucV_nm = np.zeros([map_rpmd.nbds, map_rpmd.nnuc])
        for i in range( map_rpmd.nnuc ):
            nucR_nm[:,i] = normal_mode.real_to_normal_mode( map_rpmd.nucR[:,i] )
            nucV_nm[:,i] = normal_mode.real_to_normal_mode( map_rpmd.nucP[:,i] / map_rpmd.mass[i] )

        #Evolve position of the zero-freq mode using velocity-verlet
        #this is the force on the centroid and accounts for the external force on the position
        #external force on momentum accounted for in update_nucP call of velocity-verlet algorithm
        nucR_nm[0,:] += nucV_nm[0,:] * self.delt

        #evolve position/velocities of all other modes using analytical result for hamonic oscillators
        #Note that all nuclei have same nm frequency
        c1 = np.copy( nucV_nm[1:,:] / self.nm_freq[1:,None] )
        c2 = np.copy( nucR_nm[1:,:] )
        freq_dt = self.nm_freq[1:] * self.delt

        nucR_nm[1:,:] = c1 * np.sin( freq_dt )[:,None] + c2 * np.cos( freq_dt )[:,None]

        nucV_nm[1:,:] = self.nm_freq[1:,None] * ( c1 * np.cos( freq_dt )[:,None] - c2 * np.sin( freq_dt )[:,None] )

        #Inverse transform back to real space and convert velocity back to momentum
        for i in range( map_rpmd.nnuc ):
            map_rpmd.nucR[:,i] = normal_mode.normal_mode_to_real( nucR_nm[:,i] )
            map_rpmd.nucP[:,i] = map_rpmd.mass[i] * normal_mode.normal_mode_to_real( nucV_nm[:,i] )

    ###############################################################

    def update_cayley_nucR( self, map_rpmd ):

        #Update nuclear position by a full time-step using Cayley integration scheme
        #See Korol, Bou-Rabee and Miller III JCP 2019

        #Transform position and velocities to normal-modes using fourier-transform
        #Note this is faster than directly diagonalizing the frequency matrix
        nucR_nm = np.zeros([map_rpmd.nbds, map_rpmd.nnuc])
        nucV_nm = np.zeros([map_rpmd.nbds, map_rpmd.nnuc])
        for i in range( map_rpmd.nnuc ):
            nucR_nm[:,i] = normal_mode.real_to_normal_mode( map_rpmd.nucR[:,i] )
            nucV_nm[:,i] = normal_mode.real_to_normal_mode( map_rpmd.nucP[:,i] / map_rpmd.mass[i] )

        #evolve position/velocity of all modes using cayley transform
        nucR_nm_copy = np.copy( nucR_nm )
        nucV_nm_copy = np.copy( nucV_nm )

        nucR_nm = ( self.nm_freq_dif[:,None] * nucR_nm_copy + self.delt * nucV_nm_copy ) / self.nm_freq_sum[:,None]

        nucV_nm = ( -self.nm_freq_prod[:,None] * nucR_nm_copy + self.nm_freq_dif[:,None] * nucV_nm_copy ) / self.nm_freq_sum[:,None]

        #Inverse transform back to real space and convert velocity back to momentum
        for i in range( map_rpmd.nnuc ):
            map_rpmd.nucR[:,i] = normal_mode.normal_mode_to_real( nucR_nm[:,i] )
            map_rpmd.nucP[:,i] = map_rpmd.mass[i] * normal_mode.normal_mode_to_real( nucV_nm[:,i] )

    ###############################################################

    def update_vv_mapRP( self, map_rpmd ):

        #Internal velocity-verlet algorithm for mapping variables
        #Run for small_dt_ratio number of steps for each nuclear update

        small_dt = self.delt / self.small_dt_ratio
        for _ in range( self.small_dt_ratio ):

            #Calculate first time derivative of mapping position and momentum
            d_mapR = map_rpmd.get_timederiv_mapR()
            d_mapP = map_rpmd.get_timederiv_mapP()

            #Calculate second time derivative of mapping position
            d2_mapR = map_rpmd.get_2nd_timederiv_mapR( d_mapP )

            #Update mapping position by 1/2 a time-step
            self.update_vv_mapR( map_rpmd, d_mapR, d2_mapR, small_dt )

            #Update mapping momentum by 1/4 a time-step
            self.update_vv_mapP( map_rpmd, d_mapP, small_dt )

            #Update first time derivative of mapping position at new 1/2 time-step
            d_mapP = map_rpmd.get_timederiv_mapP()

            #Update mapping momentum to another 1/4 time-step to reach 1/2 a time-step
            self.update_vv_mapP( map_rpmd, d_mapP, small_dt )

    ###############################################################

    def update_vv_mapP( self, map_rpmd, d_mapP, small_dt ):

        #Update mapping momentum by 1/4 a time-step
        map_rpmd.mapP += 0.25 * small_dt * d_mapP

    ###############################################################

    def update_vv_mapR( self, map_rpmd, d_mapR, d2_mapR, small_dt ):

        #Update nuclear position by 1/2 a time-step
        map_rpmd.mapR += 0.5 * d_mapR * small_dt + 1.0/8.0 * d2_mapR * small_dt**2

    ###############################################################

    def abm( self, map_rpmd, step ):

        #Integrates the equations of motion using a 4th order Adams-Bashforth-Moulton Predictor-Corrector integrator
        #Following equations in https://en.wikipedia.org/wiki/Linear_multistep_method
        #Predictor step is the Adams-Bashforth method for Y_n+4 to get current time derivative
        #Corrector step is the Adams-Moulton method for Y_n+3 with f(t_n+3, Y_n+3) as the result from the Predictor step

        if step < 4:
            #use rk4 integration for the first 4 steps to obtain an initial set of previous derivative values
            #store current state of system
            prev_nucR = np.copy( map_rpmd.nucR )
            prev_nucP = np.copy( map_rpmd.nucP )
            prev_mapR = np.copy( map_rpmd.mapR )
            prev_mapP = np.copy( map_rpmd.mapP )

            #propagate by a single timestep using rk4
            self.rk4( map_rpmd )

            #calculate derivative over last time step and store in previous arrays
            self.prev_d_nucR[step] = ( map_rpmd.nucR - prev_nucR ) / self.delt
            self.prev_d_nucP[step] = ( map_rpmd.nucP - prev_nucP ) / self.delt
            self.prev_d_mapR[step] = ( map_rpmd.mapR - prev_mapR ) / self.delt
            self.prev_d_mapP[step] = ( map_rpmd.mapP - prev_mapP ) / self.delt

        else:
            #Use the previous 4 derivatives to calculate the current derivative
            #store current state of system
            prev_nucR = np.copy( map_rpmd.nucR )
            prev_nucP = np.copy( map_rpmd.nucP )
            prev_mapR = np.copy( map_rpmd.mapR )
            prev_mapP = np.copy( map_rpmd.mapP )

            #Predictor step propagation
            map_rpmd.nucR = prev_nucR + ( self.prev_d_nucR[3]*55 - self.prev_d_nucR[2]*59 + self.prev_d_nucR[1]*37 - self.prev_d_nucR[0]*9 ) * self.delt / 24
            map_rpmd.nucP = prev_nucP + ( self.prev_d_nucP[3]*55 - self.prev_d_nucP[2]*59 + self.prev_d_nucP[1]*37 - self.prev_d_nucP[0]*9 ) * self.delt / 24
            map_rpmd.mapR = prev_mapR + ( self.prev_d_mapR[3]*55 - self.prev_d_mapR[2]*59 + self.prev_d_mapR[1]*37 - self.prev_d_mapR[0]*9 ) * self.delt / 24
            map_rpmd.mapP = prev_mapP + ( self.prev_d_mapP[3]*55 - self.prev_d_mapP[2]*59 + self.prev_d_mapP[1]*37 - self.prev_d_mapP[0]*9 ) * self.delt / 24

            #Shift back previous derivatives
            self.prev_d_nucR[:-1] = self.prev_d_nucR[1:]
            self.prev_d_nucP[:-1] = self.prev_d_nucP[1:]
            self.prev_d_mapR[:-1] = self.prev_d_mapR[1:]
            self.prev_d_mapP[:-1] = self.prev_d_mapP[1:]

            #Calculate and store new current derivatives for corrector step
            self.prev_d_nucR[3], self.prev_d_nucP[3], self.prev_d_mapR[3], self.prev_d_mapP[3] = map_rpmd.get_timederivs()

            #Corrector step propagation
            map_rpmd.nucR = prev_nucR + ( self.prev_d_nucR[3]*9 + self.prev_d_nucR[2]*19 - self.prev_d_nucR[1]*5 + self.prev_d_nucR[0]*1 ) * self.delt / 24
            map_rpmd.nucP = prev_nucP + ( self.prev_d_nucP[3]*9 + self.prev_d_nucP[2]*19 - self.prev_d_nucP[1]*5 + self.prev_d_nucP[0]*1 ) * self.delt / 24
            map_rpmd.mapR = prev_mapR + ( self.prev_d_mapR[3]*9 + self.prev_d_mapR[2]*19 - self.prev_d_mapR[1]*5 + self.prev_d_mapR[0]*1 ) * self.delt / 24
            map_rpmd.mapP = prev_mapP + ( self.prev_d_mapP[3]*9 + self.prev_d_mapP[2]*19 - self.prev_d_mapP[1]*5 + self.prev_d_mapP[0]*1 ) * self.delt / 24

            #Calculate and store new (updated) current derivatives for next timestep
            self.prev_d_nucR[3], self.prev_d_nucP[3], self.prev_d_mapR[3], self.prev_d_mapP[3] = map_rpmd.get_timederivs()

    ###############################################################

    def error_check( self, map_rpmd ):

        if( self.intype not in ['vv', 'analyt', 'cayley', 'rk4', 'abm'] ):
            print("ERROR: intype not one of valid types: 'vv', 'analyt', 'cayley', 'rk4', 'abm'")
            exit()

        if( self.intype in ['vv', 'analyt', 'cayley' ] and map_rpmd.methodname in ['MV-RPMD', 'mod-MV-RPMD'] ):
            print("ERROR: Cannot run", map_rpmd.methodname,"with velocity-verlet based integrator", self.intype)
            exit()

    ###############################################################

