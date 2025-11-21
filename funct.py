import matplotlib.pyplot as plt
from astropy import units as u
from mw_plot import MWSkyMap
import numpy as np

def chocolate_milkyway(center, radius, title, filename):
    """
    Generate a Milky Way sky map view with custom center and radius
    
    Parameters:

    center : tuple
        (RA, Dec) in arcsec,
    radius : tuple
        (width, height) in arcsec,
    title : str
        Title for the plot. This was to inform what picture was shown (like the andromeda)
    filename : str
        Name to save the image
    """
    
                                       ### Unpack center tuple
    ra, dec = center 
    
                                       ### Create the sky map (taken from attached file)
    mw = MWSkyMap(
        center=(ra, dec) * u.arcsec,
        radius=radius * u.arcsec,
        background="Mellinger color optical survey",
    )
    
    fig, ax = plt.subplots(figsize=(5, 5))
    mw.transform(ax)
    mw.title = title
    
    plt.savefig(filename)
    plt.show()
    print(f"Saved: {filename}")
    
# Define the conversion function (taken from attached file)
def plt2rgbarr(fig):
    """
    A function to transform a matplotlib to a 3d rgb np.array 

    Input
    -----
    fig: matplotlib.figure.Figure
        The plot that we want to encode.        

    Output
    ------
    np.array(ndim, ndim, 3): A 3d map of each pixel in a rgb encoding (the three dimensions are x, y, and rgb)
    
    """
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    rgba_buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h, w, 4))
    return rgba_arr[:, :, :3]