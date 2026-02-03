import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, AutoMinorLocator

plt.rcParams['font.family'] = 'Arial'
#plt.rcParams['font.stretch'] = 'condensed'

def gaussian(x, x0, gamma_g):
    return (1 / (gamma_g * np.sqrt(2 * np.pi))) * np.exp(-(x - x0)**2 / (2 * gamma_g**2))

def lorentzian(x, x0, gamma):
    return gamma / np.pi / ((x - x0)**2 + gamma**2)

def pseudovoigt(x, x0, gamma, weight):
    gaussian = (1 / (gamma * np.sqrt(2 * np.pi))) * np.exp(-(x - x0)**2 / (2 * gamma**2))
    lorentzian = gamma / np.pi / ((x - x0)**2 + gamma**2)
    return (1 - weight) * gaussian + weight * lorentzian

def voigt(x, x0, gamma, gamma_g):
    gaussian = (1 / (gamma_g * np.sqrt(2 * np.pi))) * np.exp(-(x - x0)**2 / (2 * gamma_g**2))
    lorentzian = gamma / np.pi / ((x - x0)**2 + gamma**2)
    return gaussian * lorentzian

def broaden_spectrum(stick_positions, stick_intensities, gamma, x_range, num_points=1000, lineshape="lorentzian", weight=1.0, gamma_g=1.0):
    x = np.linspace(x_range[0], x_range[1], num_points)
    broadened_spectrum = np.zeros_like(x)
    individual_contributions = []

    for pos, intensity in zip(stick_positions, stick_intensities):
        if lineshape == "lorentzian":
            contribution = intensity * lorentzian(x, pos, gamma)
        elif lineshape == "pseudo-voigt":
            contribution = intensity * pseudovoigt(x, pos, gamma, weight)
        elif lineshape == "voigt":
            contribution = intensity * voigt(x, pos, gamma, gamma_g)
        elif lineshape == "gaussian":
            contribution = intensity * gaussian(x, pos, gamma_g)
        else:
            raise ValueError(f"Unknown lineshape: {lineshape}")

        broadened_spectrum += contribution
        individual_contributions.append(contribution)

    return x, broadened_spectrum, individual_contributions

def normalize_to_peak(x, y, xmin, xmax):
    mask = (x >= xmin) & (x <= xmax)
    if not np.any(mask):
        raise ValueError("Normalization window does not overlap experimental data")
    return y / np.max(y[mask])

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
    parser.add_argument("--lineshape", choices=["pseudo-voigt", "lorentzian", "voigt", "gaussian"], default="lorentzian", help="Lineshape to use")
    parser.add_argument("--contributions", action="store_true", help="Plot individual stick contributions")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for stick intensities")
    parser.add_argument("--shift", type=float, default=0.0, help="Energy shift for stick positions")
    parser.add_argument("--threshold", type=float, default=0.0, help="Threshold for stick intensities included in spectrum evaluation, e.g. 0.1 is 10% of maximum transition, dominant peaks only")
    parser.add_argument("--exp", type=str, help="Experimental data file")
    parser.add_argument("--save", type=str, help="File path/filename of figure")
    parser.add_argument("--exp_norm_window", nargs=2, type=float, help="Energy window [min max] to normalize experimental spectrum")
    parser.add_argument("--legend", action="store_true", help="Show legend on plot")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--cm_inv_units",
        action="store_true",
        help="Input energies and widths are given in cm^-1"
    )
    group.add_argument(
        "--eV_units",
        action="store_true",
        help="Input energies and widths are given in eV"
    )
    args = parser.parse_args()

    conv = 8065.54429

    if args.cm_inv_units:
        to_internal = 1.0 /conv
        from_internal = conv
        unit_label = "cm$^{-1}$"
    else:
        to_internal = 1.0
        from_internal = 1.0
        unit_label = "eV"

    data = np.loadtxt(args.input_file, skiprows=1)
    stick_positions = np.array(data[:, 2], dtype=float)
    stick_intensities = data[:, 3] * args.scale

    args.shift *= to_internal
    args.x_min *= to_internal
    args.x_max *= to_internal
    args.plot_xmin *= to_internal
    args.plot_xmax *= to_internal
    args.gamma *= to_internal
    args.gamma_g *= to_internal

    plot_shift = args.shift

    max_intensity = np.max(stick_intensities)
    threshold_ = args.threshold * max_intensity
    mask = stick_intensities >= threshold_
    stick_positions = stick_positions[mask]
    stick_intensities = stick_intensities[mask]
    print(f"Threshold kept {np.sum(mask)} / {len(mask)} sticks")
    if len(stick_intensities) > 0:
        min_i = np.min(stick_intensities)
        max_i = np.max(stick_intensities)
        print(
            f"Intensity range after threshold: "
            f"min = {min_i:.4g} ({100*min_i/max_intensity:.1f}%), "
            f"max = {max_i:.4g} ({100*max_i/max_intensity:.1f}%)"
        )
    else:
        print("No sticks remain after thresholding.")

    x, spectrum, individual_contributions = broaden_spectrum(
        stick_positions, stick_intensities, args.gamma, (args.x_min, args.x_max),
        args.num_points, args.lineshape, args.weight, gamma_g=args.gamma_g if args.lineshape in ["voigt", "pseudo-voigt", "gaussian"] else None)

    x = x + plot_shift

    label_text = (
        f"gamma_l = {args.gamma * from_internal:.2f} {unit_label},\n"
        f"shift = {plot_shift * from_internal:.1f} {unit_label}"
    )

    if args.lineshape in ["voigt", "pseudo-voigt", "gaussian"]:
        label_text += f",\n gamma_g = {args.gamma_g * from_internal} {unit_label}"

    x_plot = x * from_internal
    stick_plot = (stick_positions + plot_shift) * from_internal
    plt.plot(
        x_plot,
        spectrum / np.max(spectrum),
        linewidth=1,
        color="red",
        label="Theory"
    )

    for pos, intensity in zip(stick_plot, stick_intensities):
        plt.plot([pos, pos], [0, intensity], color="red")

    if args.contributions:
        for i, contribution in enumerate(individual_contributions):
            plt.plot(x, contribution, linestyle='--', linewidth=1, label=f"Contribution {i + 1}")

    exp_offset = 0.


    if args.exp:
        exp_data = np.loadtxt(args.exp, skiprows=1)
        exp_positions_cm = exp_data[:, 0]
        exp_intensities = exp_data[:, 1] 
        exp_positions_ev = exp_positions_cm / conv

    # Convert to internal eV
    exp_positions_ev = exp_positions_cm / conv

    # Normalize in INTERNAL units
    if args.exp_norm_window:
        xmin, xmax = args.exp_norm_window
        xmin *= to_internal
        xmax *= to_internal

        exp_intensities = normalize_to_peak(
            exp_positions_ev,
            exp_intensities,
            xmin,
            xmax
        )
    else:
        exp_intensities /= np.max(exp_intensities)

    # Convert back to plotting units
    exp_positions_plot = exp_positions_ev * from_internal

    plt.plot(
        exp_positions_plot,
        exp_intensities + exp_offset,
        label="Experiment",
        color="black",
        linewidth=1
    ) 

    plt.xlabel(f"Energy / {unit_label}")
    plt.ylabel("Relative Intensity")

    if args.legend:
        plt.legend(loc='upper right', frameon=False)

    plt.ylim([0.0, 1.1])
    plt.xlim([args.plot_xmin * from_internal, args.plot_xmax * from_internal])

#Turn off yaxis
    plt.yticks([])

    ax = plt.gca()

# --- X-axis formatting (10,000 style) ---
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{x:,.0f}")
    )

# --- Tick style ---
    ax.tick_params(
        axis='both',
        which='major',
        direction='out',
        length=5,
        width=1,
        top=False,
        right=True
    )

    ax.tick_params(
        axis='both',
        which='minor',
        direction='out',
        length=3,
        width=0.8,
        top=False,
        right=True
    )

# --- Minor ticks ---
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

# --- Spine thickness (journal style) ---
    for spine in ax.spines.values():
        spine.set_linewidth(1)

# --- Remove y-axis numbers (keep axis line) ---
    ax.set_yticks([])


    if args.save:
        plt.savefig(args.save, dpi=1000)
        print(f"Figure saved to {args.save}")


    plt.show()

if __name__ == "__main__":
    main()

