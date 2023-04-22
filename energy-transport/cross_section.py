import numpy as np
import matplotlib.pyplot as plt

def cross_section(R, L, F_C, show_every=20, sanity=False, savefig=False, title=None):
    """
    plot cross section of star
    :param R: radius, array
    :param L: luminosity, array
    :param F_C: convective flux, array
    :param show_every: plot every <show_every> steps
    :param sanity: boolean, True/False
    :param savefig: boolean, True/False
    """

    # R_sun = 6.96E8      # [m]
    R_sun = R[0]
    # L_sun = 3.846E26    # [W]
    L_sun = L[0]

    plt.figure(figsize=(800/100, 800/100))
    fig = plt.gcf()
    ax  = plt.gca()

    r_range = 1.2 * R[0] / R_sun
    rmax    = np.max(R)

    ax.set_xlim(-r_range, r_range)
    ax.set_ylim(-r_range, r_range)
    ax.set_aspect('equal')

    core_limit = 0.995 * L_sun

    j = 0
    for k in range(0,len(R)-1):
        j += 1
        # plot every <show_every> steps
        if j%show_every == 0:
            if L[k] >= core_limit:     # outside core
                if F_C[k] > 0.0:       # plot convection outside core
                    circle_red = plt.Circle((0, 0), R[k]/rmax, color='red', fill=True)
                    ax.add_artist(circle_red)
                else:                  # plot radiation outside core
                    circle_yellow = plt.Circle((0, 0), R[k]/rmax, color='yellow', fill=True)
                    ax.add_artist(circle_yellow)
            else:                      # inside core
                if F_C[k] > 0.0:       # plot convection inside core
                    circle_blue = plt.Circle((0, 0), R[k]/rmax, color='blue', fill=True)
                    ax.add_artist(circle_blue)
                else:                  # plot radiation inside core
                    circle_cyan = plt.Circle((0, 0), R[k]/rmax, color='cyan', fill=True)
                    ax.add_artist(circle_cyan)
    circle_white = plt.Circle((0, 0), R[-1]/rmax, color='white', fill=True)
    ax.add_artist(circle_white)

    # create legends
    circle_red    = plt.Circle((2*r_range, 2*r_range), 0.1*r_range, color='red', fill=True)
    circle_yellow = plt.Circle((2*r_range, 2*r_range), 0.1*r_range, color='yellow', fill=True)
    circle_blue   = plt.Circle((2*r_range, 2*r_range), 0.1*r_range, color='blue', fill=True)
    circle_cyan   = plt.Circle((2*r_range, 2*r_range), 0.1*r_range, color='cyan', fill=True)

    ax.legend([circle_red,circle_yellow,circle_cyan,circle_blue],\
              ['Convection outside core','Radiation outside core','Radiation inside core','Convection inside core'])
    plt.xlabel('$R$')
    plt.ylabel('$R$')

    if title is not None:

        plt.title('Cross section of star with ' + title)

    else:

        plt.title('Cross section of star')

    if savefig:
        if sanity:
            fig.savefig('figures/sanity/sanity_cross_section.png', dpi=300)
            fig.savefig('figures/sanity/sanity_cross_section.pdf', dpi=300)
        else:
            if title is not None:
                fig.savefig('figures/testing/cross-sections/cross-'+title+'.png')
                fig.savefig('figures/testing/cross-sections/cross-'+title+'.pdf')
            else:
                fig.savefig('figures/final_cross_section.png', dpi=300)
                fig.savefig('figures/final_cross_section.pdf', dpi=300)