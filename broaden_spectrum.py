import numpy as np
import argparse
import matplotlib.pyplot as plt

#defining all broadening functions:
def lorentzian(x, x0, gamma):
    return gamma / np.pi / ((x - x0)**2 + gamma**2)

def gaussian(x, x0, gamma_g):
    return (1 / (gamma_g * np.sqrt(2 * np.pi))) * np.exp(-(x - x0)**2 / (2 * gamma_g**2))

def pseudovoigt(x, x0, gamma, weight):
    return (1 - weight) * gaussian(x, x0, gamma) + weight * lorentzian(x, x0, gamma)

def voigt(x, x0, gamma, gamma_g):
    return gaussian(x, x0, gamma_g) * lorentzian(x, x0, gamma)

def convert_units(array, mode):
    factor = 8065.54429  # 1 eV = 8065.54429 cm⁻¹
    if mode == "ev_to_cm":
        return array * factor
    elif mode == "cm_to_ev":
        return array / factor
    else:
        raise ValueError(f"Unknown conversion mode: {mode}")

# broaden stick spectra using selected lineshape
def broaden_spectrum(stick_positions, stick_intensities, gamma, x_range, num_points=1000, lineshape="lorentzian", weight=1.0, gamma_g=None):
    x = np.linspace(x_range[0], x_range[1], num_points)
    broadened_spectrum = np.zeros_like(x)
    individual_contributions = []

    for pos, intensity in zip(stick_positions, stick_intensities):
        if lineshape == "lorentzian":
            contribution = intensity * lorentzian(x, pos, gamma)
        elif lineshape == "gaussian":
            contribution = intensity * gaussian(x, pos, gamma)
        elif lineshape == "pseudo-voigt":
            contribution = intensity * pseudovoigt(x, pos, gamma, weight)
        elif lineshape == "voigt":
            contribution = intensity * voigt(x, pos, gamma, gamma_g)
        else:
            raise ValueError(f"Unknown lineshape: {lineshape}")
        broadened_spectrum += contribution
        individual_contributions.append(contribution)

    return x, broadened_spectrum, individual_contributions

def main():
    parser = argparse.ArgumentParser(description="Broaden stick spectra with specified lineshape.")
    parser.add_argument("input_file", type=str, help="Path to input file with energy and intensity columns")
    parser.add_argument("--gamma_g", type=float, default=1.0, help="Gaussian FWHM of the broadening")
    parser.add_argument("--gamma", type=float, default=1.0, help="Lorentzian FWHM of the broadening")
    parser.add_argument("--weight", type=float, default=0.5, help="Pseudo-Voigt weight of Lorentzian vs. Gaussian")
    parser.add_argument("--x_min", type=float, default=0.0, help="Minimum x-value for the spectrum range")
    parser.add_argument("--plot_xmin", type=float, default=0.0, help="Minimum x-value for the plot x-range")
    parser.add_argument("--x_max", type=float, default=50.0, help="Maximum x-value for the spectrum range")
    parser.add_argument("--plot_xmax", type=float, default=0.0, help="Maximum x-value for the plot x-range")
    parser.add_argument("--num_points", type=int, default=1000, help="Number of points in the output spectrum")
    parser.add_argument("--lineshape", choices=["lorentzian", "gaussian", "pseudo-voigt", "voigt"], default="lorentzian", help="Lineshape to use")
    parser.add_argument("--convert", choices=["ev_to_cm", "cm_to_ev"], help="Convert energy units from eV to cm⁻¹ or vice versa")
    parser.add_argument("--contributions", action="store_true", help="Plot individual stick contributions")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for stick intensities")
    parser.add_argument("--shift", type=float, default=0.0, help="Energy shift for stick positions")
    parser.add_argument("--threshold", type=float, default=0.0, help="Threshold for stick intensities included in spectrum evaluation")
    parser.add_argument("--exp", type=str, help="Experimental data file")
    parser.add_argument("--save", type=str, help="File path/filename of figure")

    args = parser.parse_args()

    data = np.loadtxt(args.input_file, skiprows=1)
    stick_positions = data[:, 2] + args.shift
    stick_intensities = data[:, 3] * args.scale

    if args.convert:
        stick_positions = convert_units(stick_positions, args.convert)
        args.x_min = convert_units(np.array([args.x_min]), args.convert)[0]
        args.x_max = convert_units(np.array([args.x_max]), args.convert)[0]
        args.plot_xmin = convert_units(np.array([args.plot_xmin]), args.convert)[0]
        args.plot_xmax = convert_units(np.array([args.plot_xmax]), args.convert)[0]
        args.shift = convert_units(np.array([args.shift]), args.convert)[0]
        args.gamma = convert_units(np.array([args.gamma]), args.convert)[0]
        if args.gamma_g:
            args.gamma_g = convert_units(np.array([args.gamma_g]), args.convert)[0]

    max_intensity = np.max(stick_intensities)
    threshold_ = args.threshold * max_intensity
    mask = stick_intensities >= threshold_
    stick_positions = stick_positions[mask]
    stick_intensities = stick_intensities[mask]

    x, spectrum, individual_contributions = broaden_spectrum(
        stick_positions, stick_intensities, args.gamma,
        (args.x_min, args.x_max), args.num_points, args.lineshape, args.weight,
        gamma_g=args.gamma_g if args.lineshape in ["voigt", "pseudo-voigt"] else None)

    unit_label = "cm⁻¹" if args.convert == "ev_to_cm" else "eV"
    label_text = f"gamma_l = {args.gamma:.2f} {unit_label}, shift = {args.shift:.2f} {unit_label}"
    if args.lineshape in ["voigt", "pseudo-voigt", "gaussian"]:
        label_text += f", gamma_g = {args.gamma_g:.2f} {unit_label}"

    plt.plot(x, spectrum / np.max(spectrum), label=label_text, linewidth=2.5, color='black')

    if args.contributions:
        for i, contribution in enumerate(individual_contributions):
            plt.plot(x, contribution, linestyle='--', label=f"Contribution {i + 1}")

    for pos, intensity in zip(stick_positions, stick_intensities):
        plt.plot([pos, pos], [0, intensity], color='#6c3483', linestyle='-')

    if args.exp:
        exp_data = np.loadtxt(args.exp, skiprows=1)
        exp_positions = exp_data[:, 0]
        exp_intensities = exp_data[:, 1] / np.max(exp_data[:, 1])
        if args.convert:
            exp_positions = convert_units(exp_positions, args.convert)
        plt.plot(exp_positions, exp_intensities, label="Experiment", linestyle='--', color='red', linewidth=1.5)

    plt.xlabel(f"Energy ({unit_label})")
    plt.ylabel("Intensity")
    plt.legend(loc='upper right')
    plt.ylim([0.0, 1])
    plt.xlim([args.plot_xmin, args.plot_xmax])

    if args.save:
        plt.savefig(args.save, dpi=300)
        print(f"Figure saved to {args.save}")
    plt.show()

if __name__ == "__main__":
    main()

