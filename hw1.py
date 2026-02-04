import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Given / required parameters
# ----------------------------
Rb = 32e9                 # 32 Gbps (assume 1 bit = 1 symbol here)
T = 1 / Rb                # symbol slot (seconds)
beta = 0.1                # roll-off / excess bandwidth factor
lam0 = 1552e-9            # 1552 nm carrier wavelength (for labeling/context)

c = 299792458.0
# carrier frequency (Hz) can be computed as c/lam0 if needed (context)

# ----------------------------
# Helper: raised cosine pulse (impulse response)
# h(t) = (1/T) * sinc(t/T) * cos(pi*beta*t/T) / (1 - (2*beta*t/T)^2)
# with special-case at t = +/- T/(2 beta)
# ----------------------------
def raised_cosine_pulse(t, T, beta):
    """
    Raised cosine (RC) pulse (Nyquist) impulse response.
    Uses communications sinc: sinc(x) = sin(pi x)/(pi x) -> np.sinc(x).
    """
    x = t / T
    denom = 1 - (2 * beta * x)**2

    # main expression
    h = (1 / T) * np.sinc(x) * np.cos(np.pi * beta * x) / denom

    # handle the removable singularities at x = +/- 1/(2 beta)
    # Using the known limit: h = (pi/(4T)) * sinc(1/(2 beta))
    if beta > 0:
        """
        We use 1/2*beta instead of T/(2*beta) because x = t/T
        because the raiased cosine formula is expressed in terms of x = t/T 
        normalized time.
        """
        x0 = 1 / (2 * beta)
        # find points close to +/- x0
        # tiny tolerence is eps to avoid floating point issues
        eps = 1e-12
        
        # mask creates a boolean to identify the two special points 
        mask = np.isclose(np.abs(x), x0, atol=eps, rtol=0)
        # if mask is met, use alternative equation 
        h[mask] = (np.pi / (4 * T)) * np.sinc(1 / (2 * beta))

    return h

# ----------------------------
# Choose a time grid:
# - need fine resolution (~1 ps or better)
# - and enough window to show the pulse shape
# ----------------------------
dt = 0.5e-12              # 0.5 ps time resolution (meets the ~1 ps guidance)
span_T = 8                # show +/- 8 symbol slots worth of time
t = np.arange(-span_T*T, span_T*T + dt, dt) # satisfies temporal resolution 

# ----------------------------
# Build an initial RC pulse, then scale so FWHM = T
# ----------------------------
p0 = raised_cosine_pulse(t, T, beta)
p0 = p0 / np.max(np.abs(p0))   # normalize peak to 1 for easy FWHM measurement

# Numerically measure FWHM of p0
half = max(p0) / 2
above = np.where(np.abs(p0) >= half)[0] # indices where pulse >= half max
t_fwhm0 = t[above[-1]] - t[above[0]]  # seconds

# Scale time so that FWHM becomes exactly T:
# If p(t) = p0(t/s), then widths scale by factor s.
s = T / t_fwhm0
p = raised_cosine_pulse(t / s, T, beta)
p = p / np.max(np.abs(p))       # normalize again

"""
Normalization takes place to allow for easier calculation of FWHM
after scaling. The pulse shape is not affected by this normalization. 
Only the vertical scale is changed. 
"""

# Confirm FWHM after scaling (optional print)
above2 = np.where(np.abs(p) >= half)[0]
t_fwhm = t[above2[-1]] - t[above2[0]]
print(f"Symbol slot T = {T*1e12:.3f} ps")
print(f"Scaled pulse FWHM = {t_fwhm*1e12:.3f} ps")

# ----------------------------
# FFT to get spectrum vs frequency offset (baseband)
# ----------------------------
N = len(t)
"""
Fast fourier transform (FFT) to convert from discrete time to frequency 
to get the spectrum of the pulse.
Frequency axis corresponding to FFT bins (Hz), then shift zero to center. 

Use fftfreq since FFT output itself has no physical frequency information, 
and fftfreq provides the mapping from FFT bin index to frequency value 
based on sampling interval dt and number of points N.

f and p sets the stage for calculating the frequency spectrum of the pulse p(t).

fftshift shifts the 0-frequency component to the center of the spectrum for
better visualization. 
"""
f = np.fft.fftfreq(N, d=dt) 
P = np.fft.fft(p)
f_shift = np.fft.fftshift(f)
P_shift = np.fft.fftshift(P)

# Convert to GHz offset from carrier (we are plotting offset, so just f_shift in GHz)
df_GHz = f_shift / 1e9

# Magnitude and phase
# abs of a vector is sqrt(real^2 + imag^2) = magnitude
mag = np.abs(P_shift)
mag = mag / np.max(mag)  # normalize for plotting

"""
np.angle gives the phase angle (in radians) of each complex number in P_shift. 
Passes the real and imaginary parts to arctan2 to compute the angle.

np.unwrap is used to remove discontinuities in the phase plot. 
Ensures a smooth phase curve by staying with a 2pi bounds.
"""
phase = np.unwrap(np.angle(P_shift))

sum_p = np.sum(p)
P0 = P[0]
print("sum(p) =", sum_p)
print("P[0]   =", P0)
print("difference =", P0 - sum_p)

# ----------------------------
# Plots (Part a deliverables)
# ----------------------------

# 1) Time-domain pulse
# Compute and plot the temporal shape of a single optical pulse
plt.figure()
plt.plot(t*1e12, p)
plt.xlabel("Time t (ps)")
plt.ylabel("Normalized amplitude")
plt.title(f"P3a: Raised-cosine pulse in time (Rb=32 Gbps, beta={beta}, FWHM=T≈{T*1e12:.2f} ps)")
plt.grid(True)

# 2) Spectrum magnitude
# Compute and plot the spectral content … plot amplitude (mag) … 
# use a frequency axis centered on the carrier and denote frequency offset … in GHz.
plt.figure()
plt.plot(df_GHz, mag)
plt.xlabel("Frequency offset Δf from carrier (GHz)")
plt.ylabel("Normalized |P(Δf)|")
plt.title("P3a: Spectrum magnitude")
plt.grid(True)
plt.xlim(-200, 200)  # adjust if you want a different view

# 3) Spectrum phase
# plot both amplitude and phase
plt.figure()
plt.plot(df_GHz, phase)
plt.xlabel("Frequency offset Δf from carrier (GHz)")
plt.ylabel("Unwrapped phase ∠P(Δf) (rad)")
plt.title("P3a: Spectrum phase")
plt.grid(True)
plt.xlim(-200, 200)  # adjust if you want a different view

plt.show()

# -----------------------------
# Part 3(b): Dispersion estimate
# -----------------------------
# Given / specified by HW
D_ps_nm_km = 3.3          # fiber dispersion [ps/(nm·km)]
# Note: `Rb`, `T`, `lam0`, and `c` are defined above and reused here.

# Spectral width estimate (HW suggestion)
delta_f = 1 / T           # Hz

# Convert spectral width to wavelength width
delta_lambda_m = (lam0**2 / c) * delta_f
delta_lambda_nm = delta_lambda_m * 1e9  # nm

print(f"Estimated Δλ ≈ {delta_lambda_nm:.2f} nm")

# Fiber lengths to evaluate
lengths_km = [10, 100]

for L in lengths_km:
    delta_T_ps = D_ps_nm_km * delta_lambda_nm * L
    print(f"Estimated temporal spread after {L} km: {delta_T_ps:.1f} ps")
    
    
# -----------------------------
# Part 3(c): Propagation with dispersion only
# -----------------------------   

# Use your same D, lam0, c

# Convert D to SI: s / m^2
D_SI = D_ps_nm_km * 1e-12 / (1e-9 * 1e3)   # (ps -> s) / (nm -> m) / (km -> m)

# beta2 in s^2 / m
beta2 = -(D_SI * lam0**2) / (2*np.pi*c)

print(f"beta2 = {beta2:.3e} s^2/m")

# Fiber lengths
lengths_km = [10, 100]

# Use the unshifted frequency axis f (Hz) that matches P = fft(p)
omega = 2*np.pi*f  # rad/s

# For plotting time: show both ps and bit-slot units
t_ps = t * 1e12
t_T = t / T

for L_km in lengths_km:
    L_m = L_km * 1e3

    # Dispersion transfer function
    """
    The sign difference comes from the convention used in the Fourier transform. 
    Textbook have it positive while NumPy uses negative in its FFT implementation.
    """
    H = np.exp(-1j * 0.5 * beta2 * L_m * (omega**2))

    # Apply in frequency domain, return to time
    P_L = P * H
    p_L = np.fft.ifft(P_L) # convert back to time domain via inverse FFT

    # Normalize magnitude for plotting
    pL_real = np.real(p_L)
    pL_real = pL_real / np.max(np.abs(pL_real))

    # -----------------------------
    # Plot temporal pulse after propagation
    # -----------------------------
    plt.figure()
    plt.plot(t_ps, pL_real)
    plt.xlabel("Time (ps)")
    plt.ylabel("Normalized amplitude (Re{p(t)})")
    plt.title(f"P3c: Time-domain pulse after {L_km} km (dispersion only)")
    plt.grid(True)

    # Same plot but x-axis in bit slots (T units)
    plt.figure()
    plt.plot(t_T, pL_real)
    plt.xlabel("Time (t/T) [bit slots]")
    plt.ylabel("Normalized amplitude (Re{p(t)})")
    plt.title(f"P3c: Time-domain pulse after {L_km} km (t/T units)")
    plt.grid(True)

    # -----------------------------
    # Frequency-domain phase: show ONLY quadratic phase
    # (remove any linear trend + unwrap)
    # -----------------------------
    # fftshift to center zero frequency for plotting
    H_shift = np.fft.fftshift(H)
    f_shift = np.fft.fftshift(f)
    df_GHz = f_shift / 1e9

    phaseH = np.unwrap(np.angle(H_shift))

    # remove best-fit linear term a*f + b (linear phase)
    """
    use f_shift for x values to fit against since it is centered around 0 Hz.
    f_shift and H_shift must stay together to maintain the correct frequency-phase relationship.
    phaseH is still H_shift.
    """
    coeff = np.polyfit(f_shift, phaseH, 1)
    # subtract linear fit from phase via m*f + b  
    phaseH_quad = phaseH - (coeff[0]*f_shift + coeff[1])

    plt.figure()
    plt.plot(df_GHz, phaseH_quad)
    plt.xlabel("Frequency offset Δf (GHz)")
    plt.ylabel("Phase (rad) [unwrapped, linear removed]")
    plt.title(f"P3c: Quadratic spectral phase after {L_km} km")
    plt.grid(True)
    plt.xlim(-200, 200)
    plt.show()
