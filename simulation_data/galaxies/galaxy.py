import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from simulation_data import get

from io import StringIO
import io

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.constants import G, h, k_B
h = 0.6774
cosmo = FlatLambdaCDM(H0= (h * 100) * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

import scipy
from scipy import stats
import scipy.spatial.KDTree

import h5py
import os
import urllib
from pathlib import Path


def get_galaxy_particle_data(id, redshift, populate_dict=False):
    '''
    input params: 
        id: the simulation id of target galaxy: integer (specific to simulation, pre-check) 
        redshift: redshift of target galaxy: numerical value (specific to simulation, pre-check at https://www.tng-project.org/data/) 
        populate_dict: boolean: False does not load dictionary of particle data (default value)
                                True loads dictionary of particle data
    preconditions: 
        requires get() imported from simulation_data.__init__
    output params: 
        checks if halo file exists. 
        if the halo file does not exist, downloads, processes and saves relevant halo properties
            if populate_dict == False: does not populate dictionary, no output
            if populate_dict == True: returns dictionary with data (6 keys)
                output dictionary keys: 'relative_x_coordinates' : x coordinates of star particles relative to the CM of the galaxy 
                                                [units: physical kpc]
                                        'relative_y_coordinates' : y coordinates of star particles relative to the CM of the galaxy 
                                                [units: physical kpc]
                                        'relative_z_coordinates' : z coordinates of star particles relative to the CM of the galaxy 
                                                [units: physical kpc]
                                        'LookbackTime' : age of star particle in lookback time
                                                [units: Lookback time in Gyr]
                                        'stellar_initial_masses' : initial stellar masses of star particles 
                                                [units: solar mass]
                                        'stellar_masses' : current stellar masses of star particles
                                                [units: solar mass]
                                        'stellar_metallicities' : current metallicities of star particles
                                                [units: solar metallicity] 
                                        'u_band' : rest-frame U band magnitude of star particles
                                                [units: Vega magnitudes]
                                        'v_band' : rest-frame V band magnitude of star particles
                                                [units: Vega magnitudes]
                                        'i_band' : rest-frame I band magnitude of star particles
                                                [units: AB magnitudes]
                                        'ParticleIDs': ids of star particles in simulation
                                                [units: none]
                                        'halfmassrad_stars': effective radius enclosing half the stellar mass
                                                [units: physical kpc]
                                        'mass_stars': total stellar mass
                                                [units: log solar mass]
    '''
    stellar_data = {}
    
    new_saved_filename = os.path.join('redshift_'+str(redshift)+'_data', 'cutout_'+str(id)+'_redshift_'+str(redshift)+'_data.hdf5')

    if Path('redshift_'+str(redshift)+'_data\cutout_'+str(id)+'_redshift_'+str(redshift)+'_data.hdf5').is_file():
        pass
    else:
        sub, saved_filename = download_data(id, redshift)
        with h5py.File(saved_filename, mode='r') as f: #read from h5py file
            dx = f['PartType4']['Coordinates'][:,0] - sub['pos_x']
            dy = f['PartType4']['Coordinates'][:,1] - sub['pos_y']
            dz = f['PartType4']['Coordinates'][:,2] - sub['pos_z']
            starFormationTime = f['PartType4']['GFM_StellarFormationTime'][:]
            starInitialMass = f['PartType4']['GFM_InitialMass'][:]
            starMass = f['PartType4']['Masses'][:]
            starMetallicity = f['PartType4']['GFM_Metallicity'][:]
            U = f['PartType4']['GFM_StellarPhotometrics'][:,0] #Vega magnitudes
            V = f['PartType4']['GFM_StellarPhotometrics'][:,2] #Vega magnitudes
            I = f['PartType4']['GFM_StellarPhotometrics'][:,6] #AB magnitudes
            ParticleIDs = f['PartType4']['ParticleIDs'][:]

        #selecting star particles only
        dx = dx[starFormationTime>0] #ckpc/h
        dy = dy[starFormationTime>0] #ckpc/h
        dz = dz[starFormationTime>0] #ckpc/h
        starInitialMass = starInitialMass[starFormationTime>0]
        starMass = starMass[starFormationTime>0]
        starMetallicity = starMetallicity[starFormationTime>0]
        U = U[starFormationTime>0] #Vega magnitudes
        V = V[starFormationTime>0] #Vega magnitudes
        I = I[starFormationTime>0] #AB magnitudes
        ParticleIDs = ParticleIDs[starFormationTime>0]
        starFormationTime = starFormationTime[starFormationTime>0]
               
        scale_factor = a = 1.0 / (1 + redshift)
        inv_sqrt_a = a**(-1/2)
        
        #unit conversions
        dx = dx*a/h #units: physical kpc
        dy = dy*a/h #units: physical kpc
        dz = dz*a/h #units: physical kpc   
        starFormationTime = 1/starFormationTime - 1 #units:scale factor
        starFormationTime = cosmo.age(starFormationTime).value #units:Gyr
        starInitialMass = starInitialMass*1e10/h #units:solar mass
        starMass = starMass*1e10/h #units:solar mass
        Gyr_redshift = cosmo.age(redshift).value #units:Gyr
        LookbackTime = Gyr_redshift - starFormationTime #units:Gyr
        starMetallicity = starMetallicity / 0.0127 #units: solar metallicity
        
        halfmassrad_stars = sub['halfmassrad_stars']*a/h
        mass_stars = np.log10(sub['mass_stars']*1e10/h)
        
        #create new file with same filename
        new_saved_filename = os.path.join('redshift_'+str(redshift)+'_data', 'cutout_'+str(id)+'_redshift_'+str(redshift)+'_data.hdf5')
        #new_saved_filename = 'cutout_'+str(id)+'_redshift_'+str(redshift)+'_data.hdf5'
        with h5py.File(new_saved_filename, 'w') as h5f:
            #writing data
            d1 = h5f.create_dataset('relative_x_coordinates', data = dx)
            d2 = h5f.create_dataset('relative_y_coordinates', data = dy)
            d3 = h5f.create_dataset('relative_z_coordinates', data = dz)
            d4 = h5f.create_dataset('LookbackTime', data = LookbackTime)
            d5 = h5f.create_dataset('stellar_initial_masses', data = starInitialMass)
            d6 = h5f.create_dataset('stellar_metallicities', data = starMetallicity)
            d7 = h5f.create_dataset('u_band', data = U) #Vega magnitudes
            d8 = h5f.create_dataset('v_band', data = V) #Vega magnitudes
            d9 = h5f.create_dataset('i_band', data = I) #Vega magnitudes
            d10 = h5f.create_dataset('ParticleIDs', data = ParticleIDs) 
            d11 = h5f.create_dataset('stellar_masses', data = starMass)
            d12 = h5f.create_dataset('halfmassrad_stars', data = halfmassrad_stars)
            d13 = h5f.create_dataset('mass_stars', data = mass_stars)
        #close file
        #h5f.close()
    
#     url = "http://www.tng-project.org/api/TNG100-1/snapshots/z=" + str(redshift) + "/subhalos/" + str(id)
#     sub = get(url)
#     scale_factor = a = 1.0 / (1 + redshift)
#     halfmassrad_stars = sub['halfmassrad_stars']*a/h
#     mass_stars = np.log10(sub['mass_stars']*1e10/h)
    
#     with h5py.File(new_saved_filename, 'a') as f:
#         del f['halfmassrad_stars']
#         del f['mass_stars']
#         d12 = f.create_dataset('halfmassrad_stars', data = [halfmassrad_stars])
#         d13 = f.create_dataset('mass_stars', data = [mass_stars])
#         f['halfmassrad_stars'] = sub['halfmassrad_stars']*a/h
#         f['mass_stars'] = np.log10(sub['mass_stars']*1e10/h)
    
    with h5py.File(new_saved_filename, 'r') as h5f_open:
        dx = h5f_open['relative_x_coordinates'][:]
        dy = h5f_open['relative_y_coordinates'][:]
        dz = h5f_open['relative_z_coordinates'][:]
        LookbackTime = h5f_open['LookbackTime'][:]
        starInitialMass = h5f_open['stellar_initial_masses'][:]
        starMetallicity = h5f_open['stellar_metallicities'][:]
        U = h5f_open['u_band'][:]
        V = h5f_open['v_band'][:]
        I = h5f_open['i_band'][:]
        ParticleIDs = h5f_open['ParticleIDs'][:]
        stellar_masses = h5f_open['stellar_masses'][:]
        halfmassrad_stars = h5f_open['halfmassrad_stars'][()]
        mass_stars = h5f_open['mass_stars'][()]
        
    stellar_data = {
                    'relative_x_coordinates' : dx, #units: physical kpc
                    'relative_y_coordinates' : dy, #units: physical kpc
                    'relative_z_coordinates' : dz, #units: physical kpc
                    'LookbackTime' : LookbackTime, #units: Gyr
                    'stellar_initial_masses' : starInitialMass, #units: solar mass
                    'stellar_metallicities' : starMetallicity, #units: solar metallicity
                    'u_band' : U, #units: Vega magnitudes
                    'v_band' : V, #units: Vega magnitudes
                    'i_band' : I, #units: AB magnitudes
                    'ParticleIDs' : ParticleIDs, #units: none
                    'stellar_masses' : stellar_masses, #units: solar mass
                    'halfmassrad_stars' : halfmassrad_stars, #units: physical kpc
                    'mass_stars' : mass_stars #units: log solar mass
                    }
                   
    if populate_dict==False:
        return
    else:
        return stellar_data
    
    
def download_data(id, redshift):
    new_saved_filename = os.path.join('redshift_'+str(redshift)+'_data', 'cutout_'+str(id)+'_redshift_'+str(redshift)+'_rawdata.hdf5')
    url = "http://www.tng-project.org/api/TNG100-1/snapshots/z=" + str(redshift) + "/subhalos/" + str(id)
    sub = get(url) # get json response of subhalo properties
    
    if Path(new_saved_filename).is_file():
        pass
    else:
        params = None
        print('Downloading ' + url)
        saved_filename = get(url + "/cutout.hdf5",params) # get and save HDF5 cutout file
        os.rename(saved_filename, new_saved_filename)
        
    return sub, new_saved_filename


def get_stellar_assembly_data(id, redshift=2, populate_dict=False):
    '''
    input params: 
        id: the simulation id of target galaxy: integer (specific to simulation, pre-check) 
        redshift: redshift of target galaxy: numerical value (default==2, specific to simulation, pre-check at https://www.tng-project.org/data/) 
        populate_dict: boolean: False does not load dictionary of particle data (default value)
                                True loads dictionary of particle data
    preconditions: 
        requires get() imported from simulation_data.__init__
        requires output from get_galaxy_particle_data(id, redshift, populate_dict=True): halo file must exist
        requires external stellar assembly files for target redshifts
    output params:
        adds data from Stellar Assembly Files
        checks if 'MergerMassRatio' exists in halo file. if key doesn't exist, saves merger mass ratio data
            if populate_dict == False: no output
            if populate_dict == True: returns 'MergerMassRatio': 
                    The stellar mass ratio of the merger in which a given ex-situ stellar particle was accreted (if applicable). 
                    The mass ratio is measured at the time when the secondary progenitor reaches its maximum stellar mass. 
                    NOTE: this quantity was calculated also in the case of flybys, without a merger actually happening.
    '''
    #open Stellar Assembly data file for z=2
    import h5py
    import os
    new_saved_filename = os.path.join('redshift_'+str(redshift)+'_data', 'cutout_'+str(id)+'_redshift_'+str(redshift)+'_data.hdf5')
    with h5py.File(new_saved_filename, 'r') as fh:
        if 'MergerMassRatio' in fh.keys():
            pass
        else:
            with h5py.File('stars_033.hdf5', 'r') as f:
                #print(f.keys())
                stars_ParticleID = f['ParticleID'][:]
                MergerMassRatio = f['MergerMassRatio'][:]
            #open galaxy particle data file
            stellar_data = get_galaxy_particle_data(id=id, redshift=redshift, populate_dict=True)
            #access particle IDs
            ParticleIDs = stellar_data['ParticleIDs']
            #select all the stars in a chosen galaxy from accretion data files
            star_file_indices = np.where(np.in1d(stars_ParticleID, ParticleIDs))[0]
            MergerMassRatio_flag = MergerMassRatio[star_file_indices]
            with h5py.File(new_saved_filename, 'a') as h5f:
                d12 = h5f.create_dataset('MergerMassRatio', data = MergerMassRatio_flag)
    if populate_dict==False:
        return
    else:
        with h5py.File(new_saved_filename, 'r') as fh:
            MergerMassRatio = fh['MergerMassRatio'][:]
        return MergerMassRatio
    
    
def get_insitu(id, redshift):
    import h5py
    
    stellar_data = get_galaxy_particle_data(id=id, redshift=redshift, populate_dict=True)
    ParticleIDs = stellar_data['ParticleIDs']
    
    with h5py.File('stars_033.hdf5', 'r') as f:
        stars_ParticleID = f['ParticleID'][:]
        insitu = f['InSitu'][:]
        
    star_file_indices = np.where(np.in1d(stars_ParticleID, ParticleIDs))[0]
    insitu_flag = insitu[star_file_indices]
    ParticleIDs = ParticleIDs[insitu_flag==1]
    
    return insitu_flag
    

def get_star_formation_history(id, redshift, plot=False, binwidth=0.05, timerange=3.2): 
    '''
    input params: 
        id: the simulation id of target galaxy: integer (specific to simulation, pre-check) 
        redshift: redshift of target galaxy: numerical value (default==2, specific to simulation, pre-check at https://www.tng-project.org/data/) 
        plot: boolean: False does not return a line plot
                       True returns a line plot of the star formation history of the target galaxy
        binwidth: width of linear age bin for computing SFH
                [units: Gyr]
    preconditions: 
        requires output from get_galaxy_particle_data(id, redshift, populate_dict=True): halo file must exist
    output params: 
        if plot==False: bin centers: centers of age bins used to construct SFH
                            [units: Gyr]
                        SFH: stellar mass in each age bin
                            [units: solar masses]
        if plot==True: line plot of normalized SFH
                            [plt.plot(bincenters, SFH/np.sum(SFH))]
    '''
    stellar_data = get_galaxy_particle_data(id=id , redshift=redshift, populate_dict=True)
    HistWeights = stellar_data['stellar_initial_masses']
    #HistWeights = stellar_data['stellar_initial_masses']/(binwidth*1e9) #units: logMsol/yr
    LookbackTime = stellar_data['LookbackTime']
    SFH, BE = np.histogram(LookbackTime, bins=np.arange(0, timerange, binwidth), weights=HistWeights, density = True)
    #SFH, BE = np.histogram(LookbackTime, bins=np.arange(0, max(LookbackTime), binwidth), weights=HistWeights)
    bincenters = np.asarray([(BE[i]+BE[i+1])/2. for i in range(len(BE)-1)])
    if plot==False:
        return bincenters, SFH
    else:     
        plt.figure(figsize=(10,7))
        plt.plot(bincenters, SFH/np.sum(SFH), color = 'b')
        plt.title('Histogram for Lookback Times for id = ' + str(id))
        plt.xlim(0, )
        plt.ylim(0, )
        plt.xlabel("Lookback Time (Gyr)")
        plt.ylabel("$M_\odot$/yr")
        return plt.show()

    
    
def timeaverage_stellar_formation_rate(id, redshift, timescale, start=0, binwidth=0.05):
    '''
    input params: 
        id: the simulation id of target galaxy: integer (specific to simulation, pre-check) 
        redshift: redshift of target galaxy: numerical value (default==2, specific to simulation, pre-check at https://www.tng-project.org/data/) 
        timescale: length of time window for over which average SFR is calculated
                [units: Gyr]
        start: minimum lookback to which timescale is is added to get time window for calculating average SFR (default 0)
                [units: Lookback time in Gyr]
        binwidth: width of linear age bin for computing SFR (default 0.05 Gyr)
                [units: Gyr]
    preconditions: 
        requires output from get_galaxy_particle_data(id, redshift, populate_dict=True): halo file must exist
    output params:
        average SFR: average star formation rate over a specified timescale 
                [units: solar mass/year] 
    '''
    BC, SFH = get_star_formation_history(redshift = redshift, id = id, plot=False, binwidth=binwidth)
    timescale_indices = np.where((np.array(BC)<=start+timescale+BC[0])&(np.array(BC)>=start)) 
    TimeAvg_SFR = np.sum([SFH[i] for i in timescale_indices]) / len(timescale_indices[0])
        #NOTE: ceiling bin value by BC[0] to accommodate edge case of timescale=start (including 0)
    return TimeAvg_SFR



def current_star_formation_rate(id, redshift):
    '''
    input params: 
        id: the simulation id of target galaxy: integer (specific to simulation, pre-check) 
        redshift: redshift of target galaxy: numerical value (default==2, specific to simulation, pre-check at https://www.tng-project.org/data/) 
    preconditions: 
        requires get() imported from simulation_data.__init__
    output params:
        current SFR: current star formation rate read from available halo properties 
                [units: solar mass/year] 
    '''
    url = "http://www.tng-project.org/api/TNG100-1/snapshots/z=" + str(redshift) + "/subhalos/" + str(id)
    sub = get(url) # get json response of subhalo properties
    return sub['sfr']




def median_stellar_age(id, redshift):
    '''
    input params: 
        id: the simulation id of target galaxy: integer (specific to simulation, pre-check) 
        redshift: redshift of target galaxy: numerical value (default==2, specific to simulation, pre-check at https://www.tng-project.org/data/) 
    preconditions: 
        requires output from get_galaxy_particle_data(id, redshift, populate_dict=True): halo file must exist
    output params:
        median stellar age: median age of star particles in target galaxy 
                [units: Lookback time in Gyr] 
    '''
    stellar_data = get_galaxy_particle_data(id=id , redshift=redshift, populate_dict=True)
    LookbackTime = stellar_data['LookbackTime']
    return np.median(LookbackTime) #units: Gyr in Lookback time



def mean_stellar_metallicity(id, redshift):
    '''
    input params: 
        id: the simulation id of target galaxy: integer (specific to simulation, pre-check) 
        redshift: redshift of target galaxy: numerical value (default==2, specific to simulation, pre-check at https://www.tng-project.org/data/) 
    preconditions: 
        requires output from get_galaxy_particle_data(id, redshift, populate_dict=True): halo file must exist
    output params:
        mean stellar metallicity: mean metallicity of star particles in target galaxy 
                [units: solar metallicity] 
    '''
    stellar_data = get_galaxy_particle_data(id=id , redshift=redshift, populate_dict=True)
    stellar_metallicities = stellar_data['stellar_metallicities']    
    return np.mean(stellar_metallicities)



def total_stellar_mass(id, redshift):
    '''
    input params: 
        id: the simulation id of target galaxy: integer (specific to simulation, pre-check) 
        redshift: redshift of target galaxy: numerical value (default==2, specific to simulation, pre-check at https://www.tng-project.org/data/) 
    preconditions: 
        requires get() imported from simulation_data.__init__
    output params:
        total stellar mass: total stellar mass of target galaxy read from available halo properties 
                [units: log10 solar masses] 
    '''
    url = "http://www.tng-project.org/api/TNG100-1/snapshots/z=" + str(redshift) + "/subhalos/" + str(id)
    sub = get(url) # get json response of subhalo properties
    return np.log10(sub['mass_stars']*1e10/h)



def halfmass_rad_stars(id, redshift):
    '''
    input params: 
        id: the simulation id of target galaxy: integer (specific to simulation, pre-check) 
        redshift: redshift of target galaxy: numerical value (default==2, specific to simulation, pre-check at https://www.tng-project.org/data/) 
    preconditions: 
        requires get() imported from simulation_data.__init__
    output params:
        half-mass radius: half-mass radius of target galaxy read from available halo properties 
                [units: physical kpc] 
    '''
    stellar_data = get_galaxy_particle_data(id=id , redshift=redshift, populate_dict=True)
    halfmassrad_stars = stellar_data['halfmassrad_stars']    
    return halfmassrad_stars #units: pkpc



def halflight_rad_stars(id, redshift, band, bound=0.5):
    '''
    input params: 
        id: the simulation id of target galaxy: integer (specific to simulation, pre-check) 
        redshift: redshift of target galaxy: numerical value (default==2, specific to simulation, pre-check at https://www.tng-project.org/data/)
        band: choice of photometric band or mass to calculate effective size in: string
                'U': (Vega magnitude) 
                'V': (Vega magnitude) 
                'I': (AB magnitude)
                'M': (solar masses)
        bound: target fraction of quantity (light intensity, mass) enclosed to calculate radius (default 0.5: for half-light radius)
                [range (0, 1]]
    preconditions: 
        requires output from get_galaxy_particle_data(id, redshift, populate_dict=True): halo file must exist
    output params:
        'U': radius enclosing central $bound fraction of U-band intensity 
                [physical kpc] 
        'V': radius enclosing central $bound fraction of V-band intensity 
                [physical kpc] 
        'I': radius enclosing central $bound fraction of I-band intensity 
                [physical kpc]
        'M': radius enclosing central $bound fraction of stellar mass
                [physical kpc]
    '''
    stellar_data = get_galaxy_particle_data(id=id , redshift=redshift, populate_dict=True)
    dx = stellar_data['relative_x_coordinates']
    dy = stellar_data['relative_y_coordinates']
    dz = stellar_data['relative_z_coordinates']
    R = (dx**2 + dy**2 + dz**2)**(1/2)#units: physical kpc
    
    if band=='U':
        mag = stellar_data['u_band']
        flux = 10**(-0.4*mag) #flux: flux = 10**(-0.4*mag)
        zipped_lists = zip(R, flux)
        sorted_pairs = sorted(zipped_lists)

        tuples = zip(*sorted_pairs)
        R_sort, band_sort = [list(tuple) for tuple in  tuples]

        band_indices = np.where(np.cumsum(np.array(band_sort))>=bound*np.sum(np.array(band_sort)))
        halflight_rad = max(np.array(R_sort)[i] for i in band_indices)
    
    elif band=='V':
        mag = stellar_data['v_band']
        flux = 10**(-0.4*mag) #flux
        zipped_lists = zip(R, flux)
        sorted_pairs = sorted(zipped_lists)

        tuples = zip(*sorted_pairs)
        R_sort, band_sort = [list(tuple) for tuple in  tuples]

        band_indices = np.where(np.cumsum(np.array(band_sort))>=bound*np.sum(np.array(band_sort)))
        halflight_rad = max(np.array(R_sort)[i] for i in band_indices)
    
    elif band=='I':
        mag = stellar_data['i_band']
        flux = 10**(-0.4*mag) #flux
        zipped_lists = zip(R, flux)
        sorted_pairs = sorted(zipped_lists)

        tuples = zip(*sorted_pairs)
        R_sort, band_sort = [list(tuple) for tuple in  tuples]

        band_indices = np.where(np.cumsum(np.array(band_sort))>=bound*np.sum(np.array(band_sort)))
        halflight_rad = max(np.array(R_sort)[i] for i in band_indices)
    
    elif band=='M':
        mass = stellar_data['stellar_masses']
        zipped_lists = zip(R, mass)
        sorted_pairs = sorted(zipped_lists)

        tuples = zip(*sorted_pairs)
        R_sort, band_sort = [list(tuple) for tuple in  tuples]

        band_indices = np.where(np.cumsum(np.array(band_sort))>=bound*np.sum(np.array(band_sort)))
        halflight_rad = max(np.array(R_sort)[i] for i in band_indices)

    return min(halflight_rad)



def age_profile(id, redshift, n_bins=20, scatter=False):
    '''
    input params: 
        id: the simulation id of target galaxy: integer (specific to simulation, pre-check) 
        redshift: redshift of target galaxy: numerical value (default==2, specific to simulation, pre-check at https://www.tng-project.org/data/)  
        n_bins: number of percentile age bins for constructing age profile, default 20 bins
                [units: none]
        scatter: boolean: False returns binned radial age data in arrays, does not return a scatter plot with raw particle data
                          True returns a scatter plot of raw particle data overlaid with a lineplot of age profile of the target galaxy
    preconditions: 
        requires output from get_galaxy_particle_data(id, redshift, populate_dict=True): halo file must exist
    output params: 
        if plot==False: statistic: array of median stellar age in each age bin
                            [units: Lookback time in Gyr]
                        radial percentiles: array of radial percentiles
                            [units: physical kpc]
                        R_e: half-mass or effective radius of target galaxy
                            [units: physical kpc]
        if plot==True: scatter plot with raw particle data overlaid with a lineplot of age profile of the target galaxy
    '''
    stellar_data = get_galaxy_particle_data(id=id , redshift=redshift, populate_dict=True)
    LookbackTime = stellar_data['LookbackTime']
    dx = stellar_data['relative_x_coordinates']
    dy = stellar_data['relative_y_coordinates']
    dz = stellar_data['relative_z_coordinates']
    metallicity = stellar_data['stellar_metallicities']
    R = (dx**2 + dy**2 + dz**2)**(1/2)#units: physical kpc
    
    radial_percentiles = np.zeros(n_bins + 1) #N+1 for N percentiles 
    for i in range(1, (n_bins+1)):
        radial_percentiles[i] = np.percentile(R, (100/n_bins)*i) 
    R_e = np.nanmedian(R)
    statistic, bin_edges, bin_number = scipy.stats.binned_statistic(R, LookbackTime, 'median', bins=radial_percentiles)
    
    if scatter==False:
        return statistic, radial_percentiles[:-1]#, R_e#, R/R_e, LookbackTime, np.log10(metallicity)
        
    else:
        plt.figure(figsize=(10,7)) # 10 is width, 7 is height
        plt.scatter(R/R_e, LookbackTime, c=np.log10(metallicity), s=0.5, alpha=0.7)#c=np.log10(metallicity)
        plt.plot(np.array(radial_percentiles[1:]/R_e)[4:-4], np.array(statistic)[4:-4], c='black')
        plt.xlim(1e-2, )
        plt.ylim(1e-1, )
        plt.grid()
        plt.colorbar(boundaries=np.linspace(-3.1,1.1,100), label='Metallicities of Stars ($\log_{10}$ $Z_\odot$)')
        plt.title('Radial Distance vs Stellar Ages (log/log scale) with Binned Age Trend for id='+str(id))
        plt.xlabel('Normalized Radial Distance (R/$R_e$)')
        plt.ylabel('Stellar Ages in Lookback Times(Gyr)')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        return plt.show()
    

def potential(id, redshift, n_bins=None):
    a = 1.0 / (1 + redshift) # scale factor
    
    # get stellar gravitational potential
    rawdata_filename = os.path.join('redshift_'+str(redshift)+'_data', 'cutout_'+str(id)+'_redshift_'+str(redshift)+'_rawdata.hdf5')    
    with h5py.File(rawdata_filename, 'r') as f:
        starFormationTime = f['PartType4']['GFM_StellarFormationTime'][:]
        grav = f['PartType4']['Potential'][:] * a #units: km/s^2
        grav = np.log10(np.abs(grav[starFormationTime>0]))
        
    if n_bins == None:
        return grav
    else:
        # get particle data
        stellar_data = get_galaxy_particle_data(id=id , redshift=redshift, populate_dict=True)
        dx = stellar_data['relative_x_coordinates']
        dy = stellar_data['relative_y_coordinates']
        dz = stellar_data['relative_z_coordinates']
        R = (dx**2 + dy**2 + dz**2)**(1/2)#units: physical kpc
        
        # calculate statistic for profile
        radial_percentiles = np.zeros(n_bins + 1) #N+1 for N percentiles 
        for i in range(1, (n_bins+1)):
            radial_percentiles[i] = np.percentile(R, (100/n_bins)*i)
        
        statistic, bin_edges, bin_number = scipy.stats.binned_statistic(R, grav, 'median', bins=radial_percentiles)
        
        return statistic, radial_percentiles[:-1], R, grav
    

def avg_abundance(id, redshift, num, den, weight=None, radius=None):
    # get particle data
    stellar_data = get_galaxy_particle_data(id=id , redshift=redshift, populate_dict=True)
    dx = stellar_data['relative_x_coordinates']
    dy = stellar_data['relative_y_coordinates']
    dz = stellar_data['relative_z_coordinates']
    #metallicity = stellar_data['stellar_metallicities']
    R = (dx**2 + dy**2 + dz**2)**(1/2)#units: physical kpc
    
    # get radius to average within
    if radius == None:
        # effective radius
        rad_limit = stellar_data['halfmassrad_stars'] #units: physical kpc
    else:
        # user-input radius
        rad_limit = radius #units: physical kpc
        
    w = np.where((R <= rad_limit)) # radii within effective radius
    
    metals = ['hydrogen', 'helium', 'carbon', 'nitrogen', 'oxygen', 'neon', 'magnesium', 'silicon', 'iron']
    
    # get metal abundance ratio
    rawdata_filename = os.path.join('redshift_'+str(redshift)+'_data', 'cutout_'+str(id)+'_redshift_'+str(redshift)+'_rawdata.hdf5')    
    with h5py.File(rawdata_filename, 'r') as f:
        starFormationTime = f['PartType4']['GFM_StellarFormationTime'][:]
        num_metal = f['PartType4']['GFM_Metals'][:,metals.index(num)]
        den_metal = f['PartType4']['GFM_Metals'][:,metals.index(den)]
        num_metal = num_metal[starFormationTime>0] # bc R above is calculated using this filter
        den_metal = den_metal[starFormationTime>0]
    ratio = num_metal / den_metal
    ratio = ratio[w]
    
    # solar abundance ratios (from http://hyperphysics.phy-astr.gsu.edu/hbase/Tables/suncomp.html)
    solar_abundance = [71.0, 27.1, 0.40, 0.096, 0.97, 0.058, 0.076, 0.099, 0.14]
    solar_ratio = solar_abundance[metals.index(num)] / solar_abundance[metals.index(den)]
    
    big_ratio = ratio / solar_ratio
    #log_ratio = np.log10(big_ratio) # un-weighted
        
    # calculate weights
    if weight=='luminosity':
        weight = 10**(-0.4 * stellar_data['v_band']) # v-band magnitude (ignoring zero point calibration factor)
        weight = weight[w]
    else:
        pass
        
    abundance = np.log10(np.average(big_ratio, weights=weight))
    
    return abundance


def avg_particular_abundance(id, redshift, metal, weight=None, radius=None):
    # get particle data
    stellar_data = get_galaxy_particle_data(id=id , redshift=redshift, populate_dict=True)
    dx = stellar_data['relative_x_coordinates']
    dy = stellar_data['relative_y_coordinates']
    dz = stellar_data['relative_z_coordinates']
    #metallicity = stellar_data['stellar_metallicities']
    R = (dx**2 + dy**2 + dz**2)**(1/2)#units: physical kpc
    
    # get radius to average within
    if radius == None:
        # effective radius
        rad_limit = stellar_data['halfmassrad_stars'] #units: physical kpc
    else:
        # user-input radius
        rad_limit = radius #units: physical kpc
        
    w = np.where((R <= rad_limit)) # radii within effective radius
    
    metals = ['hydrogen', 'helium', 'carbon', 'nitrogen', 'oxygen', 'neon', 'magnesium', 'silicon', 'iron']
    
    # get metal abundance ratio
    rawdata_filename = os.path.join('redshift_'+str(redshift)+'_data', 'cutout_'+str(id)+'_redshift_'+str(redshift)+'_rawdata.hdf5')    
    with h5py.File(rawdata_filename, 'r') as f:
        starFormationTime = f['PartType4']['GFM_StellarFormationTime'][:]
        num_metal = f['PartType4']['GFM_Metals'][:,metals.index(metal)]
        num_metal = num_metal[starFormationTime>0] # bc R above is calculated using this filter
    num_metal = num_metal[w]
    
    # solar abundance ratios (from http://hyperphysics.phy-astr.gsu.edu/hbase/Tables/suncomp.html)
    solar_abundance = [71.0, 27.1, 0.40, 0.096, 0.97, 0.058, 0.076, 0.099, 0.14]
    solar_metal = solar_abundance[metals.index(metal)]
    
    big_ratio = num_metal / solar_metal
    #log_ratio = np.log10(big_ratio) # un-weighted
        
    # calculate weights
    if weight=='luminosity':
        weight = 10**(-0.4 * stellar_data['v_band']) # v-band magnitude (ignoring zero point calibration factor)
        weight = weight[w]
    else:
        pass
        
    abundance = np.average(big_ratio, weights=weight)
    
    return abundance

    
    
def metallicity_profile(id, redshift, n_bins=20, profile='median', weight=None, axis='distance'):
    '''
    input params: 
        id: the simulation id of target galaxy: integer (specific to simulation, pre-check) 
        redshift: redshift of target galaxy: numerical value  
        n_bins: number of percentile age bins for constructing age profile, default 20 bins
                [units: none]
        profile: what kind of profile to calculate (default 'median')
                    'median': median profile
                    'mean': average profile
        weight: option to weight the profile (default None)
                    'luminosity': weight metallicity by luminosity (V-band)
    preconditions: 
        requires output from get_galaxy_particle_data(id, redshift, populate_dict=True): halo file must exist
    output params: 
        metallicity_statistic: array of chosen profile stellar metallicity in each age bin
            [units: solar metallicity]
        radial percentiles: array of radial percentiles
            [units: physical kpc]
        R: array of radii corresponding to metallicity of each particle
            [units: physical kpc]
        metallicity: array of metallicity of each particle
    '''
    stellar_data = get_galaxy_particle_data(id=id , redshift=redshift, populate_dict=True)
    LookbackTime = stellar_data['LookbackTime']
    metallicity = stellar_data['stellar_metallicities']
    
    # calculate weights for calculation
    if weight=='luminosity':
        weight = 10**(-0.4 * stellar_data['v_band']) # v-band magnitude
    else:
        weight = np.ones(len(metallicity))
    
    # get x-axis for statistic calculation
    if axis == 'distance':
        dx = stellar_data['relative_x_coordinates']
        dy = stellar_data['relative_y_coordinates']
        dz = stellar_data['relative_z_coordinates']
        R = (dx**2 + dy**2 + dz**2)**(1/2)#units: physical kpc
        xaxis = R
    elif axis == 'potential':
        grav = potential(id, redshift, n_bins=None)
        xaxis = grav
    elif axis == 'age':
        xaxis = LookbackTime
        
#     dx = stellar_data['relative_x_coordinates']
#     dy = stellar_data['relative_y_coordinates']
#     dz = stellar_data['relative_z_coordinates']
#     R = (dx**2 + dy**2 + dz**2)**(1/2)#units: physical kpc

    # calculate statistic
    percentiles = np.zeros(n_bins + 1) #N+1 for N percentiles
    for i in range(1, (n_bins+1)):
        percentiles[i] = np.percentile(xaxis, (100/n_bins)*i) 
    if profile=='median':
        metallicity_statistic, bin_edges, bin_number = scipy.stats.binned_statistic(xaxis, metallicity, 'median', bins=percentiles)
        #metallicity_statistic, bin_edges, bin_number = scipy.stats.binned_statistic(metallicity, xaxis, 'median', bins=percentiles)
        #metallicity_statistic, bin_edges, bin_number = scipy.stats.binned_statistic(xaxis, metallicity, 'median', bins=n_bins)
    elif profile=='mean':
        product_statistic, bin_edges, bin_number = scipy.stats.binned_statistic(xaxis, metallicity * weight, 'sum', bins=percentiles) # vband metallicity sum
        weight_statistic, bin_edges, bin_number = scipy.stats.binned_statistic(xaxis, weight, 'sum', bins=percentiles)
        metallicity_statistic = product_statistic / weight_statistic

    return metallicity_statistic, percentiles[:-1], xaxis, metallicity#, R
    #return metallicity_statistic, (bin_edges[1:]+bin_edges[:-1])/2, xaxis, metallicity


def metals_profile(id, redshift, num, den, n_bins=20, profile='median', weight=None, axis='distance', only_particles=False): 
    '''
    input params: 
        id: the simulation id of target galaxy: integer (specific to simulation, pre-check) 
        redshift: redshift of target galaxy: numerical value  
        n_bins: number of percentile age bins for constructing age profile, default 20 bins
                [units: none]
        profile: what kind of profile to calculate (default 'median')
                    'median': median profile
                    'mean': average profile
        weight: option to weight the profile (default None)
                    'luminosity': weight abundance by luminosity (V-band)
    preconditions: 
        requires output from get_galaxy_particle_data(id, redshift, populate_dict=True): halo file must exist
    output params: 
        statistic: array of chosen profile abundances in each age bin
            [units: solar metallicity]
        log_ratio: un-weighted log ratio of abundances to solar abundances [num/den]
        radial percentiles: array of radial percentiles
            [units: physical kpc]
        R: array of radii corresponding to metallicity of each particle
            [units: physical kpc]
    '''
    # get particle data
    stellar_data = get_galaxy_particle_data(id=id , redshift=redshift, populate_dict=True)
    metallicity = stellar_data['stellar_metallicities']
    LookbackTime = stellar_data['LookbackTime']
    
    metals = ['hydrogen', 'helium', 'carbon', 'nitrogen', 'oxygen', 'neon', 'magnesium', 'silicon', 'iron']
    
    # get metal abundance ratio
    rawdata_filename = os.path.join('redshift_'+str(redshift)+'_data', 'cutout_'+str(id)+'_redshift_'+str(redshift)+'_rawdata.hdf5')    
    with h5py.File(rawdata_filename, 'r') as f:
        starFormationTime = f['PartType4']['GFM_StellarFormationTime'][:]
        num_metal = f['PartType4']['GFM_Metals'][:,metals.index(num)]
        den_metal = f['PartType4']['GFM_Metals'][:,metals.index(den)]
        num_metal = num_metal[starFormationTime>0] # bc R above is calculated using this filter
        den_metal = den_metal[starFormationTime>0]
    ratio = num_metal / den_metal
    
    # solar abundance ratios (from http://hyperphysics.phy-astr.gsu.edu/hbase/Tables/suncomp.html)
    solar_abundance = [71.0, 27.1, 0.40, 0.096, 0.97, 0.058, 0.076, 0.099, 0.14]
    solar_ratio = solar_abundance[metals.index(num)] / solar_abundance[metals.index(den)]
    
    big_ratio = ratio / solar_ratio
    log_ratio = np.log10(big_ratio) # un-weighted
        
    # calculate weights
    if weight=='luminosity':
        weight = 10**(-0.4 * stellar_data['v_band']) # v-band magnitude (ignoring zero point calibration factor)
    else:
        weight = np.ones(len(metallicity))
        
    # get x-axis for statistic calculation
    if axis == 'distance':
        dx = stellar_data['relative_x_coordinates']
        dy = stellar_data['relative_y_coordinates']
        dz = stellar_data['relative_z_coordinates']
        R = (dx**2 + dy**2 + dz**2)**(1/2)#units: physical kpc
        xaxis = R
    elif axis == 'potential':
        grav = potential(id, redshift, n_bins=None)
        xaxis = grav
    elif axis == 'age':
        xaxis = LookbackTime
    elif axis == 'distance_norm':
        dx = stellar_data['relative_x_coordinates']
        dy = stellar_data['relative_y_coordinates']
        dz = stellar_data['relative_z_coordinates']
        R = (dx**2 + dy**2 + dz**2)**(1/2)#units: physical kpc
        R_e = stellar_data['halfmassrad_stars']
        xaxis = R/R_e
        
    if only_particles:
        return log_ratio, xaxis
    else:
        pass
        
#     dx = stellar_data['relative_x_coordinates']
#     dy = stellar_data['relative_y_coordinates']
#     dz = stellar_data['relative_z_coordinates']
#     R = (dx**2 + dy**2 + dz**2)**(1/2)#units: physical kpc
        
    # calculate statistic for profile
    percentiles = np.zeros(n_bins + 1) #N+1 for N percentiles 
    for i in range(1, (n_bins+1)):
        percentiles[i] = np.percentile(xaxis, (100/n_bins)*i)
        
    if profile=='median':
        statistic, bin_edges, bin_number = scipy.stats.binned_statistic(xaxis, log_ratio, 'median', bins=percentiles)
    elif profile=='mean':
        product_statistic, bin_edges, bin_number = scipy.stats.binned_statistic(xaxis, big_ratio * weight, 'sum', bins=percentiles) # vband metallicity sum
        weight_statistic, bin_edges, bin_number = scipy.stats.binned_statistic(xaxis, weight, 'sum', bins=percentiles)
        statistic = np.log10(product_statistic / weight_statistic)
    
    return statistic, log_ratio, percentiles[:-1], xaxis#, R      


def metals_particle_profile(id, redshift, particles, num, den, n_bins=20, profile='median', weight=None, axis='distance'): 
    '''
    input params: 
        id: the simulation id of target galaxy: integer (specific to simulation, pre-check) 
        redshift: redshift of target galaxy: numerical value  
        n_bins: number of percentile age bins for constructing age profile, default 20 bins
                [units: none]
        profile: what kind of profile to calculate (default 'median')
                    'median': median profile
                    'mean': average profile
        weight: option to weight the profile (default None)
                    'luminosity': weight abundance by luminosity (V-band)
    preconditions: 
        requires output from get_galaxy_particle_data(id, redshift, populate_dict=True): halo file must exist
    output params: 
        statistic: array of chosen profile abundances in each age bin
            [units: solar metallicity]
        log_ratio: un-weighted log ratio of abundances to solar abundances [num/den]
        radial percentiles: array of radial percentiles
            [units: physical kpc]
        R: array of radii corresponding to metallicity of each particle
            [units: physical kpc]
    '''
    # get particle data
    stellar_data = get_galaxy_particle_data(id=id , redshift=redshift, populate_dict=True)
    metallicity = stellar_data['stellar_metallicities']
    LookbackTime = stellar_data['LookbackTime']
    
    metals = ['hydrogen', 'helium', 'carbon', 'nitrogen', 'oxygen', 'neon', 'magnesium', 'silicon', 'iron']
    
    # get metal abundance ratio
    rawdata_filename = os.path.join('redshift_'+str(redshift)+'_data', 'cutout_'+str(id)+'_redshift_'+str(redshift)+'_rawdata.hdf5')    
    with h5py.File(rawdata_filename, 'r') as f:
        starFormationTime = f['PartType4']['GFM_StellarFormationTime'][:]
        num_metal = f['PartType4']['GFM_Metals'][:,metals.index(num)]
        den_metal = f['PartType4']['GFM_Metals'][:,metals.index(den)]
        num_metal = num_metal[starFormationTime>0] # bc R above is calculated using this filter
        den_metal = den_metal[starFormationTime>0]
        num_metal = num_metal[particles==1] 
        den_metal = den_metal[particles==1]
    ratio = num_metal / den_metal
    
    # solar abundance ratios (from http://hyperphysics.phy-astr.gsu.edu/hbase/Tables/suncomp.html)
    solar_abundance = [71.0, 27.1, 0.40, 0.096, 0.97, 0.058, 0.076, 0.099, 0.14]
    solar_ratio = solar_abundance[metals.index(num)] / solar_abundance[metals.index(den)]
    
    big_ratio = ratio / solar_ratio
    log_ratio = np.log10(big_ratio) # un-weighted
        
    # calculate weights
    if weight=='luminosity':
        weight = 10**(-0.4 * stellar_data['v_band']) # v-band magnitude (ignoring zero point calibration factor)
    else:
        weight = np.ones(len(ratio))
        
    # get x-axis for statistic calculation
    if axis == 'distance':
        dx = stellar_data['relative_x_coordinates'][particles==1]
        dy = stellar_data['relative_y_coordinates'][particles==1]
        dz = stellar_data['relative_z_coordinates'][particles==1]
        R = (dx**2 + dy**2 + dz**2)**(1/2)#units: physical kpc
        xaxis = R
    elif axis == 'potential':
        grav = potential(id, redshift, n_bins=None)
        xaxis = grav
    elif axis == 'age':
        xaxis = LookbackTime
    elif axis == 'distance_norm':
        dx = stellar_data['relative_x_coordinates']
        dy = stellar_data['relative_y_coordinates']
        dz = stellar_data['relative_z_coordinates']
        R = (dx**2 + dy**2 + dz**2)**(1/2)#units: physical kpc
        R_e = stellar_data['halfmassrad_stars']
        xaxis = R/R_e
        
    # calculate statistic for profile
    percentiles = np.zeros(n_bins + 1) #N+1 for N percentiles 
    for i in range(1, (n_bins+1)):
        percentiles[i] = np.percentile(xaxis, (100/n_bins)*i)
        
    if profile=='median':
        statistic, bin_edges, bin_number = scipy.stats.binned_statistic(xaxis, log_ratio, 'median', bins=percentiles)
    elif profile=='mean':
        product_statistic, bin_edges, bin_number = scipy.stats.binned_statistic(xaxis, big_ratio * weight, 'sum', bins=percentiles) # vband metallicity sum
        weight_statistic, bin_edges, bin_number = scipy.stats.binned_statistic(xaxis, weight, 'sum', bins=percentiles)
        statistic = np.log10(product_statistic / weight_statistic)
    
    return statistic, log_ratio, percentiles[:-1], xaxis


def metals_density_profile(id, redshift, num, den, n_bins=20):
    # get particle data
    stellar_data = get_galaxy_particle_data(id=id , redshift=redshift, populate_dict=True)
    metallicity = stellar_data['stellar_metallicities']
    LookbackTime = stellar_data['LookbackTime']
    mass = stellar_data['stellar_masses']
    dx = stellar_data['relative_x_coordinates']
    dy = stellar_data['relative_y_coordinates']
    dz = stellar_data['relative_z_coordinates']
    R = (dx**2 + dy**2 + dz**2)**(1/2)#units: physical kpc
    
    metals = ['hydrogen', 'helium', 'carbon', 'nitrogen', 'oxygen', 'neon', 'magnesium', 'silicon', 'iron']
    
    # get metal abundance ratio
    rawdata_filename = os.path.join('redshift_'+str(redshift)+'_data', 'cutout_'+str(id)+'_redshift_'+str(redshift)+'_rawdata.hdf5')    
    with h5py.File(rawdata_filename, 'r') as f:
        starFormationTime = f['PartType4']['GFM_StellarFormationTime'][:]
        num_metal = f['PartType4']['GFM_Metals'][:,metals.index(num)]
        den_metal = f['PartType4']['GFM_Metals'][:,metals.index(den)]
        num_metal = num_metal[starFormationTime>0] # bc R above is calculated using this filter
        den_metal = den_metal[starFormationTime>0]
    ratio = num_metal / den_metal
    
    # solar abundance ratios (from http://hyperphysics.phy-astr.gsu.edu/hbase/Tables/suncomp.html)
    solar_abundance = [71.0, 27.1, 0.40, 0.096, 0.97, 0.058, 0.076, 0.099, 0.14]
    solar_ratio = solar_abundance[metals.index(num)] / solar_abundance[metals.index(den)]
    
    big_ratio = ratio / solar_ratio
    log_ratio = np.log10(big_ratio) # un-weighted
    
    # calculate statistic for profile
    percentiles = np.zeros(n_bins + 1) #N+1 for N percentiles 
    for i in range(1, (n_bins+1)):
        percentiles[i] = np.percentile(R, (100/n_bins)*i)
        
    # density
    density = []
    mass_statistic, bin_edges, bin_number = scipy.stats.binned_statistic(R, mass, 'sum', bins=percentiles)
    for i in range(n_bins):
        dens = (3 * mass_statistic[i]) / (4*np.pi * (percentiles[i+1]**3 - percentiles[i]**3))
        density.append(dens)
        
    # abundance
    statistic, bin_edges, bin_number = scipy.stats.binned_statistic(R, log_ratio, 'median', bins=percentiles)
    
    return statistic, log_ratio, density, R, percentiles[:-1]

def histmetals_density_profile(id, redshift, num, den, n_bins=20, young=None): 
    '''
    '''
    # get particle data
    stellar_data = get_galaxy_particle_data(id=id , redshift=redshift, populate_dict=True)
    metallicity = stellar_data['stellar_metallicities']
    LookbackTime = stellar_data['LookbackTime']
    mass = stellar_data['stellar_masses']
    dx = stellar_data['relative_x_coordinates']
    dy = stellar_data['relative_y_coordinates']
    dz = stellar_data['relative_z_coordinates']
    R = (dx**2 + dy**2 + dz**2)**(1/2)#units: physical kpc
    
    metals = ['hydrogen', 'helium', 'carbon', 'nitrogen', 'oxygen', 'neon', 'magnesium', 'silicon', 'iron']
    
    # get metal abundance ratio and other stuff
    rawdata_filename = os.path.join('redshift_'+str(redshift)+'_data', 'cutout_'+str(id)+'_redshift_'+str(redshift)+'_rawdata.hdf5')    
    with h5py.File(rawdata_filename, 'r') as f:
        starFormationTime = f['PartType4']['GFM_StellarFormationTime'][:]
        if young: # only consider stars within last 100 million years (0.1 Gyr)
            sub, saved_filename = download_data(id, redshift)
            formationTime = young[0]
            formationGyr = young[1]
#             print(formationTime, formationGyr)
#             starFormationRedshift = 1/starFormationTime - 1 #units:redshift
#             starFormationGyr = cosmo.age(starFormationRedshift).value #units:Gyr
            Gyr_redshift = cosmo.age(redshift).value #units:Gyr
#             cutoff = Gyr_redshift - 0.1 #units:Gyr
            starFormationGyr = np.interp(starFormationTime, formationTime, formationGyr)
            age = Gyr_redshift - starFormationGyr
#             print(np.min(age))
            
            flag = (starFormationTime>0)&(age<0.1)#&(age<=Gyr_redshift) # look at stars born before cutoff
            
            mass = f['PartType4']['Masses'][flag]*1e10/h
            dx = f['PartType4']['Coordinates'][:,0] - sub['pos_x']
            dy = f['PartType4']['Coordinates'][:,1] - sub['pos_y']
            dz = f['PartType4']['Coordinates'][:,2] - sub['pos_z']
            dx = dx[flag]
            dy = dy[flag]
            dz = dz[flag]
            R = (dx**2 + dy**2 + dz**2)**(1/2)
        else:
            flag = (0<starFormationTime)
        num_metal = f['PartType4']['GFM_Metals'][:,metals.index(num)]
        den_metal = f['PartType4']['GFM_Metals'][:,metals.index(den)]
        num_metal = num_metal[flag] # bc R above is calculated using this filter
        den_metal = den_metal[flag]
        stellarHsml = f['PartType4']['StellarHsml'][flag] # ckpc/h
    ratio = num_metal / den_metal
    
    # solar abundance ratios (from http://hyperphysics.phy-astr.gsu.edu/hbase/Tables/suncomp.html)
    solar_abundance = [71.0, 27.1, 0.40, 0.096, 0.97, 0.058, 0.076, 0.099, 0.14]
    solar_ratio = solar_abundance[metals.index(num)] / solar_abundance[metals.index(den)]
    
    big_ratio = ratio / solar_ratio
    log_ratio = np.log10(big_ratio) # un-weighted
    
    # calculate statistic for profile
#     percentiles = np.zeros(n_bins + 1) #N+1 for N percentiles 
#     for i in range(1, (n_bins+1)):
#         percentiles[i] = np.percentile(R, (100/n_bins)*i)
        
    # density
    scale_factor = a = 1.0 / (1 + redshift)
    stellarHsml = stellarHsml*a/h # physical kpc
    density = (mass*32) / (4/3*np.pi*stellarHsml**3)

    return log_ratio, density, R
    
    
def gasmetals_profile(id, redshift, num, den, n_bins=20, profile='median', weight=None, axis='distance'): 
    '''
    input params: 
        id: the simulation id of target galaxy: integer (specific to simulation, pre-check) 
        redshift: redshift of target galaxy: numerical value  
        n_bins: number of percentile age bins for constructing age profile, default 20 bins
                [units: none]
        profile: what kind of profile to calculate (default 'median')
                    'median': median profile
                    'mean': average profile
        weight: option to weight the profile (default None)
                    'luminosity': weight abundance by luminosity (V-band)
    preconditions: 
        requires output from get_galaxy_particle_data(id, redshift, populate_dict=True): halo file must exist
    output params: 
        statistic: array of chosen profile abundances in each age bin
            [units: solar metallicity]
        log_ratio: un-weighted log ratio of abundances to solar abundances [num/den]
        radial percentiles: array of radial percentiles
            [units: physical kpc]
        R: array of radii corresponding to metallicity of each particle
            [units: physical kpc]
    '''    
    url = "http://www.tng-project.org/api/TNG100-1/snapshots/z=" + str(redshift) + "/subhalos/" + str(id)
    sub = get(url) # get json response of subhalo properties
    if sub['mass_gas'] > 0:
        pass
    else:
        return [1], [1], [1], [1]
    
    metals = ['hydrogen', 'helium', 'carbon', 'nitrogen', 'oxygen', 'neon', 'magnesium', 'silicon', 'iron']
    
    # get metal abundance ratio
    rawdata_filename = os.path.join('redshift_'+str(redshift)+'_data', 'cutout_'+str(id)+'_redshift_'+str(redshift)+'_rawdata.hdf5')    
    with h5py.File(rawdata_filename, 'r') as f:
        num_metal = f['PartType0']['GFM_Metals'][:,metals.index(num)]
        den_metal = f['PartType0']['GFM_Metals'][:,metals.index(den)]
        dx = f['PartType0']['Coordinates'][:,0] - sub['pos_x']
        dy = f['PartType0']['Coordinates'][:,1] - sub['pos_y']
        dz = f['PartType0']['Coordinates'][:,2] - sub['pos_z']
    ratio = num_metal / den_metal
    
    # solar abundance ratios (from http://hyperphysics.phy-astr.gsu.edu/hbase/Tables/suncomp.html)
    solar_abundance = [71.0, 27.1, 0.40, 0.096, 0.97, 0.058, 0.076, 0.099, 0.14]
    solar_ratio = solar_abundance[metals.index(num)] / solar_abundance[metals.index(den)]
    
    big_ratio = ratio / solar_ratio
    log_ratio = np.log10(big_ratio) # un-weighted
        
    # calculate weights
    if weight==None:
        weight = np.ones(len(num_metal))
        
    # get x-axis for statistic calculation    
    if axis == 'distance':
        a = 1.0 / (1 + redshift)
        h = 0.6774            
            
        dx = dx*a/h #units: physical kpc
        dy = dy*a/h #units: physical kpc
        dz = dz*a/h #units: physical kpc   
        
        R = (dx**2 + dy**2 + dz**2)**(1/2)#units: physical kpc
        xaxis = R
        
    # calculate statistic for profile
    percentiles = np.zeros(n_bins + 1) #N+1 for N percentiles 
    for i in range(1, (n_bins+1)):
        percentiles[i] = np.percentile(xaxis, (100/n_bins)*i)
        
    if profile=='median':
        statistic, bin_edges, bin_number = scipy.stats.binned_statistic(xaxis, log_ratio, 'median', bins=percentiles)
    elif profile=='mean':
        product_statistic, bin_edges, bin_number = scipy.stats.binned_statistic(xaxis, big_ratio * weight, 'sum', bins=percentiles) # vband metallicity sum
        weight_statistic, bin_edges, bin_number = scipy.stats.binned_statistic(xaxis, weight, 'sum', bins=percentiles)
        statistic = np.log10(product_statistic / weight_statistic)
    
    return statistic, log_ratio, percentiles[:-1], xaxis


def max_merger_ratio(id, redshift=2, scale=30):
    '''
    input params: 
        id: the simulation id of target galaxy: integer (specific to simulation, pre-check) 
        redshift: redshift of target galaxy: numerical value (default==2, specific to simulation, pre-check at https://www.tng-project.org/data/)  
        scale: the radial distance up to which star partickes should be considered as part of the target galaxy (default 30 kpc)
                [units: physical kpc]
    preconditions: 
        requires output from get_galaxy_particle_data(id, redshift, populate_dict=True): halo file must exist
        requires output from get_stellar_assembly_data(id, redshift, populate_dict=True): assembly data must exist
    output params: 
        greatest merger contribution: the largest fraction of current stellar mass within $scale [kpc] that can be traced back to a single merger event
                [units: none, fraction of total stellar mass within $scale [kpc] from galaxy CM]
    '''
    MergerMassRatio = get_stellar_assembly_data(id=id, redshift=redshift, populate_dict=True)
    stellar_data = get_galaxy_particle_data(id=id, redshift=redshift, populate_dict=True)
    dx = stellar_data['relative_x_coordinates']
    dy = stellar_data['relative_y_coordinates']
    dz = stellar_data['relative_z_coordinates']
    R = (dx**2 + dy**2 + dz**2)**(1/2)    
    stellar_masses = stellar_data['stellar_masses'][R<=scale]
    MergerMassRatio = MergerMassRatio[R<=scale]
    R = R[R<=scale]
    
    unique_MMR = np.asarray(list(set(MergerMassRatio)))
    MMR = unique_MMR[unique_MMR>0]

    TM = np.zeros(0)
    for x in MMR:
        TM = np.concatenate((TM, np.sum(stellar_masses[MergerMassRatio==x])), axis = None)
    
    return max(TM)/np.sum(stellar_masses)

