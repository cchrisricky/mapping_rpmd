    def run_PIMD( self, Nsteps=None, resample=None, intype=None, Nprint=100, delt=None, init_time=0.0, small_dt_ratio=1 ):

        #routine to run PIMC only for the nuclei
        #using normal modes to get initial thermal configuration

        #resample - the number of steps that equilibrate the temperature (NVT MD)
        
        #Initialize the integrator
        self.integ = integrator.integrator( self, delt, intype, small_dt_ratio )

        #converting to string, taking the length, and subtracting 2 (for the 0.) gives us the length of just the decimal component
        tDigits = len( str( math.modf(Nprint * delt)[0] ) ) - 2

        #Automatically initialize nuclear momentum from MB distribution if none have been specified
        if( self.nucP is None ):
            print('Automatically initializing nuclear momentum to Maxwell-Boltzmann distribution at beta_p =', self.beta_p*self.nbds ,' / ',self.nbds)
            self.get_nucP_MB()

        #Initialize nuclear positions to zero if None specified
        if( self.nucR is None ):
            print('Automatically initializing all nuclear positions to zero')
            self.nucR = np.zeros([self.nbds, self.nnuc])

        #Open output files
        self.file_output = open( 'output.dat', 'w' )
        self.file_nucR   = open( 'nucR.dat','w' )
        self.file_nucP   = open( 'nucP.dat','w' )

        print()
        print( '#########################################################' )
        print('running equilibrium PIMC for', Nsteps, 'at beta_p =,' self.beta_p*self.nbds, ' / ',self.nbds)
        print( '#########################################################' )
        print()

        current_time = init_time
        step = 0
        for step in range( Nsteps ):

            #resample the nucP if we need to
            if(resample != None):
                if( np.mod( step, resample ) == 0):
                    print('resample the bead momentum to run within an NVT ensemble')
                    self.get_nucP_MB()

            #Print data starting with initial time
            if( np.mod( step, Nprint ) == 0 ):
                print('Writing data at step', step, 'and time', format(current_time, '.'+str(tDigits)+'f'), 'for PIMD calculation')
                self.print_PIMD_data( current_time )
                sys.stdout.flush()

            #Outer loop to integrate EOM using velocity-verlet like algorithms
            #This includes pengfei's implementation, and the analtyical and cayley modification of it

            #If initial step of dynamics need to initialize electronic hamiltonian
            #and derivative of nuclear momentum (aka the force on the nuclei)
            #NOTE: moving forward these calls may need to be generalized to allow for other types of methods
            if( step == 0 ):
                #self.potential.calc_Hel( self.nucR )
                if( self.intype == 'vv' ):
                    self.integ.d_nucP_for_vv = self.potential.calc_state_indep_force( self.nucR ) + self.potential.calc_rp_harm_force( self.nucR, self.beta_p, self.mass )
                else:
                    self.integ.d_nucP_for_vv = self.potential.calc_state_indep_force( self.nucR )

            #Update nuclear momentum by 1/2 a time-step
            self.integ.update_vv_nucP( self )

            #Update mapping variables by 1/2 a time-step
            #self.update_vv_mapRP( self )

            #Update nuclear position for full time-step
            if( self.intype == 'vv' ):
                self.integ.update_vv_nucR( self )
            elif( self.intype == 'analyt' ):
                self.integ.update_analyt_nucR( self )
            elif( self.intype == 'cayley' ):
                self.integ.update_cayley_nucR( self )

            #Update electronic Hamiltonian given new positioninteg
            #Update mapping variables to full time-step
            #self.update_vv_mapRP( self )

            #Calculate derivative of nuclear momentum at new time-step (aka the force on the nuclei)
            #Don't include contribution from internal modes of ring-polymer if doing analyt or cayley
            if( self.intype == 'vv' ):
                self.integ.d_nucP_for_vv = self.potential.calc_state_indep_force( self.nucR ) + self.potential.calc_rp_harm_force( self.nucR, self.beta_p, self.mass )
            else:
                self.integ.d_nucP_for_vv = self.potential.calc_state_indep_force( self.nucR )

            #Update nuclear momentum to full time-step
            self.integ.update_vv_nucP( self )

            #Increase current time
            current_time = init_time + delt * (step+1)

        #Print data at final step regardless of Nprint
        print('Writing data at step ', step+1, 'and time', format(current_time, '.'+str(tDigits)+'f'), 'for PIMD Dynamics calculation')
        self.print_PIMD_data( current_time )
        sys.stdout.flush()

        #Close output files
        self.file_output.close()
        self.file_nucR.close()
        self.file_nucP.close()

        print()
        print( '#########################################################' )
        print( 'END', self.methodname, 'PIMD' )
        print( '#########################################################' )
        print()
    
    #####################################################################

    def PIMD_integ(self, delt, step):

        #subroutine to propagate nucP and nucR using velocity velet

        #calculate the derivatives of nucP
        d_nucP = self.potential.calc_state_indep_force( self.nucR ) + self.potential.calc_rp_harm_force( self.nucR, self.beta_p, self.mass )

        #upate nuclear momentum by 1/2 a time-step
        self.nucP += 0.5 * self.delt * d_nucP

        #Update nuclear position for full time-step
        self.nucR += self.nucP * delt / self.mass

        #calculate the derivatives of nucP
        d_nucP = self.potential.calc_state_indep_force( self.nucR ) + self.potential.calc_rp_harm_force( self.nucR, self.beta_p, self.mass )

        #upate nuclear momentum by another 1/2 of time-step
        self.nucP += 0.5 * self.delt * d_nucP