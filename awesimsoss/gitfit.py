"""
Module to split up and reassemble large files so they can be
committed to Github repositories
"""
from glob import glob
import os
import shutil
import sys

from astropy.io import fits
import numpy as np


def disassemble(file, MB_limit=80, destination=None):
    """
    Disassemble a FITS file into smaller chunks

    Parameters
    ----------
    file: str
        The path to the large file
    MB_limit: float, int
        The maximum file size affter disassembly
    destination: str (optional)
        The destination for the disassembled files

    Returns
    -------
    list
        The list of file paths
    """
    # List of files to return
    filelist = []

    # Check file size in MB
    filesize = os.path.getsize(file) / 1000000

    # Filename
    filename = os.path.basename(file).replace('.fits', '')

    # Get destination
    if destination is None:
        destination = os.path.dirname(file)

    # If already small enough, do nothing
    if filesize > MB_limit:

        # Open the FITS file
        hdulist = fits.open(file, mode='update')

        # Strip file of data
        extensions = {}
        for hdu in hdulist:

            # Save the real data
            extensions[hdu.name] = hdu.data

            # Replace with tiny dummy array
            hdulist[hdu.name].data = None

        # Write to the file and close it
        hdulist.writeto(file, overwrite=True)
        hdulist.close()

        # Make a folder
        folder = filename + '_data'
        destination = os.path.join(destination, folder)
        os.system('mkdir {}'.format(destination))

        # Write the data to .npz files
        for ext, data in extensions.items():

            # Some are None
            if data is not None:

                # Check data size in MB
                datasize = data.nbytes

                # Get number of chunks
                nchunks = np.ceil(datasize / 1000000 / MB_limit).astype(int)

                # Break up into chunks
                chunks = np.array_split(data, nchunks + 2)

                # Save as .npz files
                for n, chunk in enumerate(chunks):

                    # Determine filename
                    chunkname = filename + '.{}.{}.npy'.format(ext, n)

                    # Save the chunk to file
                    filepath = os.path.join(destination, chunkname)
                    np.save(filepath, chunk)

                    # Add to list of filenames
                    filelist.append(filepath)

    return filelist


def make_dummy_file(file, shape=(15, 2000, 2000), n_ext=2):
    """
    Make a dummy FITS file for testing

    Parameters
    ----------
    file: str
        The path and filename to use
    shape: tuple
        The desired shape of the data
    n_ext: int
        The number of SCI extensions

    Returns
    -------
    str
        The path to the new file
    """
    # Primary HDU
    hdulist = [fits.PrimaryHDU()]

    # SCI extensions
    for n in range(n_ext):
        hdu = fits.ImageHDU(data=np.random.normal(size=shape), name='SCI_{}'.format(n))
        hdulist.append(hdu)

    # Make list
    hdulist = fits.HDUList(hdulist)

    # Write the file
    hdulist.writeto(file, overwrite=True)

    print("{} MB file created at {}".format(os.path.getsize(file) / 1000000, file))


def reassemble(file, save=False):
    """
    Reassemble a FITS file from its components

    Parameters
    ----------
    file: str
        The path to the FITS file
    save: bool
        Save the data to the file again

    Returns
    -------
    astropy.io.fits.HDUList
        The HDU list
    """
    # Open the FITS file
    hdulist = fits.open(file, mode='update')
    filename = os.path.basename(file).replace('.fits', '')
    directory = os.path.join(os.path.dirname(file), filename + '_data')

    # Large file
    if os.path.isdir(directory):

        # Populate file with data
        for hdu in hdulist:

            # Get the real data files
            filestr = filename + '.{}.*'.format(hdu.name)
            files = glob(os.path.join(directory, filestr))

            # Load and recombine the data
            if len(files) > 0:
                data = np.concatenate([np.load(f) for f in files])
            else:
                data = None

            # Replace with real data
            hdulist[hdu.name].data = data

        # Write the file changes
        if save:
            hdulist.writeto(file, overwrite=True)
            shutil.rmtree(directory)

    return hdulist
