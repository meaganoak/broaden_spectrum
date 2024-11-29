import numpy as np
import argparse
import matplotlib.pyplot as plt

# Lorentzian broadening function
def lorentzian(x, x0, gamma):
    """
    Lorentzian function for broadening
    Parameters:
    x (array): Array of energies
    x0 (float): Center position of the funciton
    gamma (float): Full-width at half maximum (FWHM)

    Returns:
    array: Broadening of a single stick at x0.
    """
    return gamma / np.pi / ((x - x0)**2 + gamma**2)

# Pseudo-Voigt function
def pseudovoigt(x, x0, gamma, weight):
    """
    Pseudo-Voigt function for broadening
    Parameters:
    x (array): Array of energies
    x0 (float): Center position of the function
    gamma (float): Full-width at half maximum (FWHM)
    weight (float): weighting of Lorentzian vs. Gaussian

    Returns:
    array: Broadening of a single stick at x0.
    """
    gaussian = (1 / (gamma * np.sqrt(2 * np.pi))) * np.exp(-(x - x0)**2 / (2 * gamma**2))
    lorentzian = gamma / np.pi / ((x - x0)**2 + gamma**2)
    return (1 - weight) * gaussian + weight * lorentzian


# Broaden stick spectra with individual contributions
def broaden_spectrum(stick_positions, stick_intensities, gamma, x_range, num_points=1000, lineshape="lorentzian", weight=1.0):
    """
    Broadens stick spectra using a specified lineshape.

    Parameters:
    stick_positions (array): Positions of the stick spectra.
    stick_intensities (array): Intensities of the stick spectra.
    gamma (float): FWHM of the lineshape.
    x_range (tuple): Range of x-values to calculate (min, max).
    num_points (int): Number of points for the output x-axis.
    lineshape (str): Type of lineshape ("lorentzian" or "pseudo-voigt").
    weight (float): Pseudo-Voigt weight of Lorentzian vs. Gaussian.

    Returns:
    tuple: x-axis, total broadened spectrum, and individual contributions.
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    broadened_spectrum = np.zeros_like(x)
    individual_contributions = []

    for pos, intensity in zip(stick_positions, stick_intensities):
        if lineshape == "lorentzian":
            contribution = intensity * lorentzian(x, pos, gamma)
        elif lineshape == "pseudo-voigt":
            contribution = intensity * pseudovoigt(x, pos, gamma, weight)
        else:
            raise ValueError(f"Unknown lineshape: {lineshape}")

        broadened_spectrum += contribution
        individual_contributions.append(contribution)

    return x, broadened_spectrum, individual_contributions

# Main function to parse arguments and run broadening
def main():
    parser = argparse.ArgumentParser(description="Broaden stick spectra with specified lineshape.")
    parser.add_argument("input_file", type=str, help="Path to input file with energy and intensity columns")
    parser.add_argument("--gamma", type=float, default=1.0, help="FWHM of the broadening")
    parser.add_argument("--weight", type=float, default=0.5, help="Pseudo-Voigt weight of Lorentzian vs. Gaussian")
    parser.add_argument("--x_min", type=float, default=0.0, help="Minimum x-value for the spectrum range")
    parser.add_argument("--x_max", type=float, default=50.0, help="Maximum x-value for the spectrum range")
    parser.add_argument("--num_points", type=int, default=1000, help="Number of points in the output spectrum")
    parser.add_argument("--lineshape", choices=["pseudo-voigt", "lorentzian"], default="lorentzian", help="Lineshape to use")
    parser.add_argument("--contributions", action="store_true", help="Plot individual stick contributions")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for stick intensities")
    parser.add_argument("--shift", type=float, default=0.0, help="Energy shift for stick positions")

    args = parser.parse_args()

    data = np.loadtxt(args.input_file)
    stick_positions = data[:, 0] + args.shift
    stick_intensities = data[:, 1]*args.scale

    x, spectrum, individual_contributions = broaden_spectrum(
        stick_positions, stick_intensities, args.gamma, (args.x_min, args.x_max),
        args.num_points, args.lineshape, args.weight)

    #Plotting broadened spectrum
    plt.plot(x, spectrum, label="Total Broadened Spectrum", linewidth=2)

    #Contributions flag turned on for individual contributions from each transition 
    if args.contributions:
        for i, contribution in enumerate(individual_contributions):
            plt.plot(x, contribution, linestyle='--', label=f"Contribution {i + 1}")

    # Plot the stick spectrum - should add flag to turn this off if required
    for pos, intensity in zip(stick_positions, stick_intensities):
        plt.plot([pos, pos], [0, intensity], 'r--', label="Stick Spectrum" if pos == stick_positions[0] else "")

    plt.xlabel("Energy")
    plt.ylabel("Intensity")
    plt.legend()
    plt.ylim([0.0, 1])  # Adjust as needed
    plt.xlim([3722, 3738])
    plt.show()

if __name__ == "__main__":
    main()
