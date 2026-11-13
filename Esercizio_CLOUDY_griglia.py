from pathlib import Path
import re
import subprocess
import numpy as np
import pandas as pd
from scipy.optimize import root
from scipy.special import kv

c = 2.99792458e10
G = 6.67430e-8
h = 6.62607015e-27
kB = 1.380649e-16
sigma_sb = 5.670374419e-5
Msun = 1.98847e33
RYD_ERG = 13.605693122994 * 1.602176634e-12

ROOT = Path.cwd()
CLOUDY_EXE = Path('/home/maddaba/c25.00/source/cloudy.exe')
OUTPUT_DIR = ROOT / 'griglia_cloudy_runs'
RESULT_FILE = ROOT / 'luminosita_teoriche_BLR.txt'
MASSES = [1e8, 5e8, 1e9]  # masse solari
MDOTS = [1e-3, 5e-2, 5e-1, 1e-1]  # dimensionless
LOG_N_H = 10.0
LOG_RADIUS = 18.0
LOG_COLUMN = 23.0
COVERING_FACTOR = 0.1
# Attivare solo se verificato che il comando non scala gia' le intensita .int.
APPLY_CF_TO_INTENSITY = False          #SEMBRA SIA GI° PRESENTE FACENDO TEST !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
# Sigma non definita dal modello: NaN, non errore osservativo.

# Ogni riga e' una singola transizione, non un blend automatico.
# Le lunghezze sono in Angstrom, ricercate entro una tolleranza esplicita.
LINES = {
    'Halpha': ('H', 1, 6564.0),
    'Hbeta': ('H', 1, 4861.0),
    'Hdelta': ('H', 1, 4102.0),
    'CIII': ('C', 3, 1909.0),
    'CIV': ('C', 4, 1549.0),     ######################################################ERRORE 1549.0   ma in realtà era giusto e poi messo tolleranza a 0.001 --> mi da corretto ma luminosità viene 0
    'HeII': ('He', 2, 228.855),   ######################################################ERRORE 228.85  --> ok se messo a 228.855 e tolleranza 0.001
    'MgII': ('Mg', 2, 2798.0),   ######################################################ERRORE 2798.0  --> ok se messo a 2798 e tolleranza 0.001
}
# I valori richiesti sono approssimati: NON selezionare un multipletto ambiguo    #RIVEDERE
TOLERANCE_A = {'Halpha': 2.0, 'Hbeta': 1.0, 'Hdelta': 1.0,
               'CIII': 0.5, 'CIV': 0.001, 'HeII': 0.001, 'MgII': 0.001}    # PRIMA ERANO :HeII=0.2, MgII=3.0, CIV=1.0

CFG = {
    'eta': 0.083, 'Nr': 80, 'Nnu_ss': 120,
    'nu_min_ss': 1e13, 'nu_max_ss': 1e17,
    'alfa': 0.3, 'rmin': 3.0, 'rmax': 1000.0,
    'delta': 0.0, 'beta_adaf': 0.5,
    'c1': 0.5, 'c3': 0.3, 'tnorm': 1.68e-10,
    'Nnu_adaf': 120, 'nu_min_adaf': 1e10, 'nu_max_adaf': 1e17,
    'root_y0': [0.5, 1.0, 0.99], 'root_tol': 1e-10,
    'logmdot_switch': -2.0,
}

def build_ss_spectrum(
    black_hole_mass_solar,
    eddington_accretion_rate,
    cfg
):
    """
    Compute the spectral energy distribution of a
    Shakura-Sunyaev thin accretion disk.

    Input:
        - black_hole_mass_solar:
          black-hole mass in solar masses

        - eddington_accretion_rate:
          accretion rate normalized to the Eddington value

        - cfg:
          configuration dictionary containing numerical
          and physical model parameters

    Processing:
        - compute Eddington luminosity
        - derive physical mass accretion rate
        - construct radial disk grid
        - compute disk temperature profile
        - evaluate blackbody emission at each radius
        - integrate emission over the entire disk

    Output:
        - frequency array
        - monochromatic luminosity spectrum (Lnu)
        - spectral energy distribution (nuLnu)
        - dictionary containing model information
    """

    # Convert black-hole mass from solar masses to CGS units
    black_hole_mass = black_hole_mass_solar * Msun

    # Retrieve model parameters from configuration
    radiative_efficiency = cfg["eta"]

    number_radial_points = cfg["Nr"]

    number_frequency_points = cfg["Nnu_ss"]

    minimum_frequency = cfg["nu_min_ss"]
    maximum_frequency = cfg["nu_max_ss"]

    # Compute Eddington luminosity
    eddington_luminosity = (
        1.3e38 * black_hole_mass_solar
    )

    # Compute total disk luminosity
    disk_luminosity = (
        eddington_accretion_rate
        * eddington_luminosity
    )

    # Convert luminosity into physical mass accretion rate
    mass_accretion_rate = (
        disk_luminosity
        / (radiative_efficiency * c**2)
    )

    # Compute characteristic disk radii
    schwarzschild_radius = (
        2 * G * black_hole_mass
        / c**2
    )

    inner_radius = 3 * schwarzschild_radius

    outer_radius = 3000 * schwarzschild_radius

    # Construct logarithmically spaced radial grid
    radius_grid = np.logspace(
        np.log10(inner_radius),
        np.log10(outer_radius),
        number_radial_points
    )

    # Inner-boundary correction factor
    boundary_factor = (
        1.0
        - np.sqrt(inner_radius / radius_grid)
    )

    boundary_factor = np.clip(
        boundary_factor,
        0.0,
        None
    )

    # Compute effective temperature profile
    temperature_profile = (
        (
            3
            * G
            * black_hole_mass
            * mass_accretion_rate
        )
        /
        (
            8
            * np.pi
            * sigma_sb
            * radius_grid**3
        )
        * boundary_factor
    )**0.25

    # Construct frequency grid
    frequency_grid = np.logspace(
        np.log10(minimum_frequency),
        np.log10(maximum_frequency),
        number_frequency_points
    )

    # Allocate Planck-function matrix
    planck_function = np.zeros(
        (
            number_frequency_points,
            number_radial_points
        )
    )

    # Evaluate blackbody emission at each radius
    for j in range(number_radial_points):

        if temperature_profile[j] <= 0:
            continue

        x = (
            h * frequency_grid
            /
            (
                kB
                * temperature_profile[j]
            )
        )

        # Prevent numerical overflow
        x = np.clip(
            x,
            1e-12,
            700.0
        )

        planck_function[:, j] = (
            (
                2
                * h
                * frequency_grid**3
                / c**2
            )
            /
            (
                np.exp(x) - 1.0
            )
        )

    # Integrate the contribution of all disk annuli
    integrand = (
        radius_grid[None, :]
        * planck_function
    )

    integrated_intensity = np.trapz(
        integrand,
        radius_grid,
        axis=1
    )

    # Convert intensity into luminosity
    luminosity_spectrum = (
        4                                        
        * np.pi**2
        * integrated_intensity
    )

    # Compute spectral energy distribution
    spectral_energy_distribution = (
        frequency_grid
        * luminosity_spectrum
    )

    # Store additional physical information
    model_info = {

        "regime": "SS",

        "black_hole_mass_solar":
            black_hole_mass_solar,

        "eddington_accretion_rate":
            eddington_accretion_rate,

        "eddington_luminosity":
            eddington_luminosity,

        "disk_luminosity":
            disk_luminosity,

        "mass_accretion_rate_cgs":
            mass_accretion_rate
    }

    return (
        frequency_grid,
        luminosity_spectrum,
        spectral_energy_distribution,
        model_info
    )

def bessel_k(order, argument):
    """
    Compute the modified Bessel function of the second kind.

    Input:
        - order: order of the Bessel function
        - argument: value at which the function is evaluated

    Output:
        - value of the modified Bessel function
    """

    return kv(order, argument)

def adaf_thermal_function(theta_e):
    """
    Compute the ADAF thermal correction function.

    Input:
        - theta_e: dimensionless electron temperature

    Processing:
        - evaluate the Bessel function required by the ADAF model
        - check that the result is finite and physically valid
        - compute the thermal correction term

    Output:
        - thermal correction factor
    """

    bessel_value = bessel_k(2, 1.0 / theta_e)

    if not np.isfinite(bessel_value) or bessel_value <= 0:
        return np.nan

    return (
        (1.0 / bessel_value)
        * (2.0 + 2.0 * theta_e + 1.0 / theta_e)
        * np.exp(-1.0 / theta_e)
    )

def adaf_temperature_equations(
    variables,
    accretion_rate,
    black_hole_mass,
    beta,
    cfg
):
    """
    Define the nonlinear ADAF equations solved by the root finder.

    Input:
        - variables: numerical variables explored by the solver
        - accretion_rate: Eddington-scaled accretion rate
        - black_hole_mass: black-hole mass in solar masses
        - beta: gas-to-total pressure parameter
        - cfg: configuration dictionary

    Processing:
        - transform solver variables into physical quantities
        - compute heating and cooling terms
        - compute synchrotron, Compton, and bremsstrahlung contributions
        - return the residuals of the ADAF balance equations

    Output:
        - list of three residuals to be driven close to zero
    """

    # Reject invalid numerical inputs
    if not np.all(np.isfinite(variables)):
        return [1e99, 1e99, 1e99]

    # Reject non-physical model parameters
    if (
        accretion_rate <= 0
        or black_hole_mass <= 0
        or beta <= 0
        or beta >= 1
    ):
        return [1e99, 1e99, 1e99]

    # Avoid extreme solver excursions
    if np.any(np.abs(variables) > 1e3):
        return [1e99, 1e99, 1e99]

    alpha = cfg["alfa"]
    inner_radius = cfg["rmin"]
    outer_radius = cfg["rmax"]
    electron_heating_fraction = cfg["delta"]
    c1 = cfg["c1"]
    c3 = cfg["c3"]
    temperature_normalization = cfg["tnorm"]

    # Squared variables guarantee positive physical quantities
    advection_fraction = variables[2]**2
    synchrotron_parameter = 1e3 * variables[1]**2
    theta_e = variables[0]**2

    if (
        advection_fraction <= 0
        or synchrotron_parameter <= 0
        or theta_e <= 1e-12
    ):
        return [1e99, 1e99, 1e99]

    thermal_factor = adaf_thermal_function(theta_e)

    if not np.isfinite(thermal_factor):
        return [1e99, 1e99, 1e99]

    # ADAF scaling coefficients
    synchrotron_scale_1 = (
        1.42e9
        * alpha**(-0.5)
        * np.sqrt(1 - beta)
        * c1**(-0.5)
        * c3**0.5
    )

    synchrotron_scale_2 = (
        1.19e-13
        * synchrotron_parameter
    )

    # Ion heating term
    ion_heating = (
        1.2e4
        * thermal_factor
        * alpha**(-2)
        * c1**(-2)
        * c3
        * beta
        * black_hole_mass
        * accretion_rate**2
        / inner_radius
    )

    # Direct electron heating term
    electron_heating = (
        electron_heating_fraction
        * 9.39e4
        * ((1 - beta) / advection_fraction)
        * c3
        * black_hole_mass
        * accretion_rate
        / inner_radius
    )

    total_heating = ion_heating + electron_heating

    peak_synchrotron_frequency = (
        synchrotron_scale_1
        * synchrotron_scale_2
        * black_hole_mass**(-0.5)
        * accretion_rate**0.5
        * (theta_e / temperature_normalization)**2
        * inner_radius**(-5/4)
    )

    if (
        not np.isfinite(peak_synchrotron_frequency)
        or peak_synchrotron_frequency <= 0
    ):
        return [1e99, 1e99, 1e99]

    synchrotron_power = (
        53.0
        * (synchrotron_parameter / 1000.0)**3
        * (theta_e / (1e9 * temperature_normalization))**7
        * black_hole_mass**0.5
        * accretion_rate**1.5
    )

    compton_amplification = (
        1
        + 4 * theta_e
        + 16 * theta_e**2
    )

    optical_depth = (
        23.87
        * accretion_rate
    )

    if optical_depth <= 0 or compton_amplification <= 1:
        return [1e99, 1e99, 1e99]

    compton_slope = (
        -np.log(optical_depth)
        / np.log(compton_amplification)
    )

    try:
        compton_base = (
            6.2e7
            * (theta_e / (1e9 * temperature_normalization))
            / (peak_synchrotron_frequency * 1e-12)
        )

        if (
            compton_base <= 0
            or not np.isfinite(compton_base)
        ):
            return [1e99, 1e99, 1e99]

        compton_factor = (
            compton_base**(1 - compton_slope)
            - 1
        )

        compton_power = (
            (synchrotron_power / 0.71)
            / (1 - compton_slope)
            * compton_factor
        )

    except Exception:
        return [1e99, 1e99, 1e99]

    # Bremsstrahlung correction factor
    if theta_e < 1:
        bremsstrahlung_factor = (
            4
            * np.sqrt(2 * theta_e / np.pi**3)
            * (1 + 1.781 * theta_e**1.34)
        )

    else:
        bremsstrahlung_factor = (
            (9 * theta_e / (2 * np.pi))
            * (
                np.log(1.123 * theta_e + 0.48)
                + 1.5
            )
        )

    bremsstrahlung_power = (
        4.78
        * alpha**(-2)
        * c1**(-2)
        * np.log(outer_radius / inner_radius)
        * bremsstrahlung_factor
        * black_hole_mass
        * accretion_rate**2
    )

    # First residual: heating minus total cooling
    residual_energy = (
        total_heating
        - synchrotron_power
        - compton_power
        - bremsstrahlung_power
    )

    # Second residual: advected energy balance
    residual_advection = (
        (1 - advection_fraction)
        * 9.39e4
        * ((1 - beta) / advection_fraction)
        * c3
        * black_hole_mass
        * accretion_rate
        / inner_radius
        - total_heating
    )

    bessel_value = bessel_k(2, 1 / theta_e)
    logarithm_argument = theta_e**3 * bessel_value

    if (
        not np.isfinite(logarithm_argument)
        or logarithm_argument <= 0
    ):
        return [1e99, 1e99, 1e99]

    transformed_synchrotron_parameter = (
        synchrotron_parameter**(1/3)
    )

    # Third residual: synchrotron self-absorption condition
    residual_synchrotron = (
        transformed_synchrotron_parameter
        + 1.852 * np.log(transformed_synchrotron_parameter)
        - 10.36
        - 0.26 * np.log(black_hole_mass * accretion_rate)
        + 0.26 * np.log(logarithm_argument)
    )

    residuals = np.array(
        [
            residual_energy,
            residual_synchrotron,
            residual_advection
        ],
        dtype=float
    )

    if not np.all(np.isfinite(residuals)):
        return [1e99, 1e99, 1e99]

    return residuals.tolist()

def build_adaf_spectrum_from_solution(
    electron_temperature,
    advection_fraction,
    synchrotron_parameter,
    black_hole_mass,
    accretion_rate,
    beta,
    cfg
):
    """
    Compute the ADAF spectrum after the physical solution is found.

    Input:
        - electron_temperature: electron temperature parameter
        - advection_fraction: fraction of energy advected inward
        - synchrotron_parameter: synchrotron self-absorption parameter
        - black_hole_mass: black-hole mass in solar masses
        - accretion_rate: Eddington-scaled accretion rate
        - beta: gas-to-total pressure parameter
        - cfg: configuration dictionary

    Processing:
        - build the frequency grid
        - compute synchrotron emission
        - compute Comptonized synchrotron emission
        - compute bremsstrahlung emission
        - sum all components

    Output:
        - logarithmic frequency grid
        - logarithmic spectral energy distribution
    """

    alpha = cfg["alfa"]
    inner_radius = cfg["rmin"]
    outer_radius = cfg["rmax"]
    c1 = cfg["c1"]
    c3 = cfg["c3"]
    temperature_normalization = cfg["tnorm"]

    minimum_frequency = cfg["nu_min_adaf"]
    maximum_frequency = cfg["nu_max_adaf"]
    number_frequency_points = cfg["Nnu_adaf"]

    log_frequency_grid = np.linspace(
        np.log10(minimum_frequency),
        np.log10(maximum_frequency),
        number_frequency_points
    )

    frequency_grid = 10**log_frequency_grid

    theta_e = (
        temperature_normalization
        * electron_temperature
    )

    if electron_temperature <= 0 or theta_e <= 0:
        raise RuntimeError("Invalid ADAF temperature")

    synchrotron_scale_1 = (
        1.42e9
        * alpha**(-0.5)
        * np.sqrt(1 - beta)
        * c1**(-0.5)
        * c3**0.5
    )

    synchrotron_scale_2 = (
        1.19e-13
        * synchrotron_parameter
    )

    synchrotron_scale_3 = 1.05e-24

    compton_amplification = (
        1
        + 4 * theta_e
        + 16 * theta_e**2
    )

    optical_depth = (
        23.87
        * accretion_rate
    )

    if optical_depth <= 0 or compton_amplification <= 1:
        raise RuntimeError("Invalid ADAF Compton parameters")

    compton_slope = (
        -np.log(optical_depth)
        / np.log(compton_amplification)
    )

    log_peak_synchrotron_frequency = np.log10(
        synchrotron_scale_1
        * synchrotron_scale_2
        * black_hole_mass**(-0.5)
        * accretion_rate**0.5
        * electron_temperature**2
        * inner_radius**(-5/4)
    )

    log_maximum_compton_frequency = (
        np.log10(electron_temperature)
        + 10.796
    )

    log_synchrotron_peak_luminosity = (
        3 * np.log10(synchrotron_scale_1 * synchrotron_scale_2)
        + np.log10(synchrotron_scale_3)
        - (7/4) * np.log10(inner_radius)
        + np.log10(black_hole_mass**0.5 * accretion_rate**1.5)
        + 7 * np.log10(electron_temperature)
    )

    log_synchrotron_luminosity = np.zeros(
        number_frequency_points
    )

    log_compton_luminosity = np.zeros(
        number_frequency_points
    )

    log_bremsstrahlung_luminosity = np.zeros(
        number_frequency_points
    )

    if theta_e < 1:
        bremsstrahlung_factor = (
            4
            * np.sqrt(2 * theta_e / np.pi**3)
            * (1 + 1.781 * theta_e**1.34)
        )

    else:
        bremsstrahlung_factor = (
            (9 * theta_e / (2 * np.pi))
            * (
                np.log(1.123 * theta_e + 0.48)
                + 1.5
            )
        )

    for i in range(number_frequency_points):

        # Synchrotron component below the peak frequency
        if log_frequency_grid[i] < log_peak_synchrotron_frequency:

            log_synchrotron_luminosity[i] = (
                np.log10(synchrotron_scale_3)
                + (8/5) * np.log10(
                    synchrotron_scale_1
                    * synchrotron_scale_2
                )
                + np.log10(
                    black_hole_mass**(6/5)
                    * accretion_rate**(4/5)
                )
                + (21/5) * np.log10(electron_temperature)
                + (2/5) * log_frequency_grid[i]
            )

        # Comptonized synchrotron component above the peak
        if log_frequency_grid[i] > log_peak_synchrotron_frequency:

            compton_base = (
                log_synchrotron_peak_luminosity
                + (compton_slope - 1)
                * log_peak_synchrotron_frequency
                - compton_slope
                * log_frequency_grid[i]
            )

            compton_cutoff = (
                -(
                    10**(
                        log_frequency_grid[i]
                        - log_maximum_compton_frequency
                    )
                )
                / np.log(10)
            )

            log_compton_luminosity[i] = (
                compton_base
                + compton_cutoff
            )

        bremsstrahlung_base = (
            24.36
            - 2 * np.log10(alpha * c1)
            + np.log10(
                np.log(outer_radius / inner_radius)
                * bremsstrahlung_factor
            )
            - np.log10(electron_temperature)
            + np.log10(
                black_hole_mass
                * accretion_rate**2
            )
        )

        bremsstrahlung_cutoff = (
            -(
                4.8e-11
                * frequency_grid[i]
                / electron_temperature
            )
            / np.log(10)
        )

        log_bremsstrahlung_luminosity[i] = (
            bremsstrahlung_base
            + bremsstrahlung_cutoff
        )

    # Total ADAF emission from all components
    total_flux = (
        10**log_synchrotron_luminosity
        + 10**log_compton_luminosity
        + 10**log_bremsstrahlung_luminosity
    )

    if (
        not np.all(np.isfinite(total_flux))
        or np.any(total_flux < 0)
    ):
        raise RuntimeError("Invalid ADAF flux")

    log_spectral_energy_distribution = (
        log_frequency_grid
        + np.log10(total_flux)
    )

    return (
        log_frequency_grid,
        log_spectral_energy_distribution
    )

def compute_adaf_spectral_data(
    black_hole_mass,
    accretion_rate,
    beta,
    cfg,
    verbose=False
):
    """
    Solve the ADAF equations and build the corresponding spectrum.

    Input:
        - black_hole_mass: black-hole mass in solar masses
        - accretion_rate: Eddington-scaled accretion rate
        - beta: gas-to-total pressure parameter
        - cfg: configuration dictionary
        - verbose: if True, print the current model parameters

    Processing:
        - initialize the root solver
        - solve the ADAF nonlinear equations
        - recover the physical ADAF quantities
        - compute the spectral energy distribution
        - convert logarithmic quantities into physical luminosities

    Output:
        - frequency grid
        - monochromatic luminosity spectrum
        - spectral energy distribution
        - model information dictionary
    """

    if verbose:
        print(
            f"ADAF -> m={black_hole_mass:.3e}, "
            f"mdot={accretion_rate:.3e}, beta={beta}"
        )

    initial_solution = cfg["root_y0"]

    solution = root(
        adaf_temperature_equations,
        initial_solution,
        args=(
            accretion_rate,
            black_hole_mass,
            beta,
            cfg
        ),
        tol=cfg["root_tol"]
    )

    if not solution.success:
        raise RuntimeError(
            f"Root did not converge for "
            f"m={black_hole_mass:.3e}, "
            f"mdot={accretion_rate:.3e}"
        )

    solver_variables = solution.x

    theta_e = solver_variables[0]**2

    electron_temperature = (
        theta_e
        / cfg["tnorm"]
    )

    synchrotron_parameter = (
        1e3
        * solver_variables[1]**2
    )

    advection_fraction = (
        solver_variables[2]**2
    )

    (
        log_frequency_grid,
        log_spectral_energy_distribution
    ) = build_adaf_spectrum_from_solution(
        electron_temperature,
        advection_fraction,
        synchrotron_parameter,
        black_hole_mass,
        accretion_rate,
        beta,
        cfg
    )

    frequency_grid = 10**log_frequency_grid

    log_luminosity_spectrum = (
        log_spectral_energy_distribution
        - log_frequency_grid
    )

    luminosity_spectrum = (
        10**log_luminosity_spectrum          
    )

    spectral_energy_distribution = (
        frequency_grid
        * luminosity_spectrum
    )

    model_info = {

        "regime": "ADAF",

        "black_hole_mass":
            black_hole_mass,

        "accretion_rate":
            accretion_rate,

        "beta":
            beta,

        "electron_temperature":
            electron_temperature,

        "synchrotron_parameter":
            synchrotron_parameter,

        "advection_fraction":
            advection_fraction,

        "root_success":
            solution.success
    }

    return (
        frequency_grid,
        luminosity_spectrum,
        spectral_energy_distribution,
        model_info
    )

def build_adaf_spectrum(
    black_hole_mass_solar,
    eddington_accretion_rate,
    cfg
):
    """
    High-level wrapper for the ADAF spectrum.

    Input:
        - black_hole_mass_solar: black-hole mass in solar masses
        - eddington_accretion_rate: Eddington-scaled accretion rate
        - cfg: configuration dictionary

    Processing:
        - read the ADAF beta parameter from the configuration
        - solve the ADAF physical equations
        - remove invalid numerical values
        - sort the spectrum by increasing frequency

    Output:
        - cleaned frequency grid
        - cleaned luminosity spectrum
        - cleaned spectral energy distribution
        - ADAF model information
    """

    beta = cfg["beta_adaf"]

    (
        frequency_grid,
        luminosity_spectrum,
        spectral_energy_distribution,
        model_info
    ) = compute_adaf_spectral_data(
        black_hole_mass=black_hole_mass_solar,
        accretion_rate=eddington_accretion_rate,
        beta=beta,
        cfg=cfg,
        verbose=False
    )

    # Keep only finite and physically meaningful values
    valid_mask = (
        np.isfinite(frequency_grid)
        & np.isfinite(luminosity_spectrum)
        & np.isfinite(spectral_energy_distribution)
        & (frequency_grid > 0)
        & (luminosity_spectrum >= 0)
    )

    frequency_grid = frequency_grid[valid_mask]
    luminosity_spectrum = luminosity_spectrum[valid_mask]
    spectral_energy_distribution = (
        spectral_energy_distribution[valid_mask]
    )

    if len(frequency_grid) < 2:
        raise RuntimeError(
            "Invalid ADAF spectrum after filtering"
        )

    sorting_index = np.argsort(frequency_grid)

    return (
        frequency_grid[sorting_index],
        luminosity_spectrum[sorting_index],
        spectral_energy_distribution[sorting_index],
        model_info
    )

def build_unified_spectrum(
    log_black_hole_mass,
    log_accretion_rate,
    cfg
):
    """
    Build the continuum spectrum using either the ADAF or SS model.

    Input:
        - log_black_hole_mass: base-10 logarithm of black-hole mass
        - log_accretion_rate: base-10 logarithm of accretion rate
        - cfg: configuration dictionary

    Processing:
        - convert logarithmic parameters into physical values
        - compare the accretion rate with the regime threshold
        - use the ADAF spectrum at low accretion rate
        - use the SS spectrum at high accretion rate

    Output:
        - frequency grid
        - luminosity spectrum
        - spectral energy distribution
        - model information dictionary
    """

    black_hole_mass = 10**log_black_hole_mass
    accretion_rate = 10**log_accretion_rate

    # Select the accretion regime according to the accretion rate
    if log_accretion_rate < cfg["logmdot_switch"]:

        (
            frequency_grid,
            luminosity_spectrum,
            spectral_energy_distribution,
            model_info
        ) = build_adaf_spectrum(
            black_hole_mass_solar=black_hole_mass,
            eddington_accretion_rate=accretion_rate,
            cfg=cfg
        )

        regime = "ADAF"

    else:

        (
            frequency_grid,
            luminosity_spectrum,
            spectral_energy_distribution,
            model_info
        ) = build_ss_spectrum(
            black_hole_mass_solar=black_hole_mass,
            eddington_accretion_rate=accretion_rate,
            cfg=cfg
        )

        regime = "SS"

    model_info["regime"] = regime

    return (
        frequency_grid,
        luminosity_spectrum,
        spectral_energy_distribution,
        model_info
    )



def save_sed(nu, lnu, path):
    nu = np.asarray(nu, dtype=float)
    lnu = np.asarray(lnu, dtype=float)
    good = np.isfinite(nu) & np.isfinite(lnu) & (nu > 0) & (lnu > 0)
    nu, lnu = nu[good], lnu[good]
    ix = np.argsort(nu)
    nu, lnu = nu[ix], lnu[ix]
    if len(nu) < 3 or np.any(np.diff(nu) <= 0):
        raise ValueError('SED non valida')
    lint = float(np.trapz(lnu, nu))
    if not np.isfinite(lint) or lint <= 0:
        raise ValueError('Luminosita integrata non valida')
    with path.open('w') as f:
        f.write('# E(Ryd)  Lnu/max(Lnu)\n')
        for v, l in zip(nu, lnu):
            f.write(f'{h*v/RYD_ERG:.12e} {l/lnu.max():.12e}\n')
        f.write('***\n')
    return lint


def parse_intensita_cloudy(path):
    text = path.read_text(errors='replace')
    if not text or 'per unit inner area' not in text.splitlines()[0].lower():
        raise ValueError(f'Unita del file .int non verificate: {path}')
    if 'begin emission lines' not in text.lower():
        raise ValueError('Manca begin emission lines nel file .int')
    text = re.split(r'begin emission lines', text, flags=re.I)[-1]
    rows = []
    # L'etichetta CLOUDY puo' contenere descrittori come Inwd, Coll, ecc.
    pat = re.compile(r'^\s*([A-Za-z]{1,2})\s+(\d+)\s+(.*?)\s+(\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?[Amun])\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s*$')
    for raw in text.splitlines():
        m = pat.match(raw)
        if not m:
            continue
        element, stage, descriptor, wave, intensity = m.groups()
        # Accettiamo soltanto la riga totale senza etichette componenti.
        if descriptor.strip():
            continue
        unit = wave[-1]
        wavelength = float(wave[:-1])
        if unit == 'm':  # micron
            wavelength *= 1e4
        elif unit == 'u':  # micron, in alcune configurazioni
            wavelength *= 1e4
        elif unit == 'n':  # nm
            wavelength *= 10
        elif unit != 'A':
            continue
        rows.append((element.capitalize(), int(stage), wavelength, float(intensity)))
    if not rows:
        raise ValueError(f'Nessuna riga totale riconosciuta in {path}')
    return pd.DataFrame(rows, columns=['element', 'stage', 'wavelength_A', 'I_area'])


def extract_line(df, name, specification):                         ##########################################################
    elem, stage, wavelength = specification            # Element, stage, wavelength
    candidates = df[(df.element == elem) & (df.stage == stage)].copy()
    candidates['delta_A'] = abs(candidates.wavelength_A - wavelength)  # Compute the absolute difference in wavelength
    candidates = candidates[candidates.delta_A <= TOLERANCE_A[name]]
    if len(candidates) != 1:
        details = candidates[['wavelength_A', 'I_area']].to_dict('records')
        raise ValueError(f'{name} {wavelength} A: {len(candidates)} corrispondenze: {details}. '
                         'Verificare transizioni/blend nel .int e correggere LINES.')
    return float(candidates.iloc[0].I_area)


def run_one(mass, mdot):
    log_m, log_mdot = np.log10(mass), np.log10(mdot)
    nu, lnu, _, info = build_unified_spectrum(log_m, log_mdot, CFG)
    regime = info['regime']
    stem = f'M{mass:.3e}_mdot{mdot:.3e}_{regime}'.replace('+', 'p').replace('-', 'm')
    directory = OUTPUT_DIR / stem
    directory.mkdir(parents=True, exist_ok=True)
    lint = save_sed(nu, lnu, directory / (stem + '.sed'))
    commands = [
        f'title M={mass:.6e} mdot={mdot:.6e} {regime}',
        f'table SED "{stem}.sed"',
        f'luminosity total {np.log10(lint):.9f}',
        f'radius {LOG_RADIUS:.4f}',
        f'hden {LOG_N_H:.4f}',
        f'stop column density {LOG_COLUMN:.4f}',
        f'covering factor {COVERING_FACTOR:.6f}',
        'iterate',
        f'save last lines intensity "{stem}.int" emergent column all',
        f'save last lines labels "{stem}.labels"',
        'database print',
        'print citation',
    ]
    (directory / (stem + '.in')).write_text('\n'.join(commands) + '\n')
    run = subprocess.run([str(CLOUDY_EXE), '-r', stem], cwd=directory,
                         capture_output=True, text=True)
    out = directory / (stem + '.out')
    outtext = out.read_text(errors='replace') if out.exists() else ''
    if run.returncode != 0 or 'Cloudy exited OK' not in outtext:
        raise RuntimeError(f'CLOUDY fallito per M={mass} mdot={mdot}\n'
                           + run.stderr[-2000:] + '\n' + '\n'.join(outtext.splitlines()[-30:]))
    df = parse_intensita_cloudy(directory / (stem + '.int'))
    radius = 10.0**LOG_RADIUS
    area = 4 * np.pi * radius**2
    # Il file .int e' per unita' di superficie interna. Verificare che il comando
    # covering factor non abbia gia' incorporato Cf in questa specifica uscita.
    cf_scale = COVERING_FACTOR if APPLY_CF_TO_INTENSITY else 1.0
    rows = []
    for name, spec in LINES.items():
        try:
            intensity = extract_line(df, name, spec)
            luminosity = area * cf_scale * intensity
        except ValueError as exc:
            print('ATTENZIONE:', exc)
            luminosity = np.nan
        rows.append((mass, mdot, name, spec[2], luminosity, np.nan))
    return rows


def main():
    if not CLOUDY_EXE.is_file():
        raise FileNotFoundError(f'Eseguibile CLOUDY non trovato: {CLOUDY_EXE}')
    if not 0 < COVERING_FACTOR <= 1:
        raise ValueError('Covering factor deve essere in (0, 1]')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Sovrascrittura all'avvio, anche in caso di arresto successivo.
    with RESULT_FILE.open('w') as f:
        f.write('# massa mdot elemento_ionizzato lambda_trans luminosita sigma_lum\n')
    for mass in MASSES:
        for mdot in MDOTS:
            print(f'Avvio: M={mass:.3e} Msun, mdot={mdot:.3e}')
            try:
                rows = run_one(mass, mdot)
            except Exception as exc:
                print(f'ERRORE: M={mass:.3e} mdot={mdot:.3e}: {exc}')
                rows = [(mass, mdot, name, spec[2], np.nan, np.nan)
                        for name, spec in LINES.items()]
            with RESULT_FILE.open('a') as f:
                for m, md, name, lam, lum, sigma in rows:
                    f.write(f'{m:.8e} {md:.8e} {name} {lam:.4f} {lum:.8e} {sigma:.8e}\n')
    print(f'File salvato: {RESULT_FILE}')


if __name__ == '__main__':
    main()
