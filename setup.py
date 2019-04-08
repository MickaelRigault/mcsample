#! /usr/bin/env python
#

DESCRIPTION = "mcsample: MCMC Sample low level python tools "
LONG_DESCRIPTION = """ mcsample: MCMC Sample low level python tools """

DISTNAME = 'mcsample'
AUTHOR = 'Mickael Rigault, Martin Briday, Romain Graziani'
MAINTAINER = 'Mickael Rigault' 
MAINTAINER_EMAIL = 'm.rigault@ipnl.in2p3.fr'
URL = 'https://github.com/MickaelRigault/mcsample'
LICENSE = 'Apache 2.0'
DOWNLOAD_URL = 'https://github.com/MickaelRigault/mcsample/0.1'
VERSION = '0.1.0'

try:
    from setuptools import setup, find_packages
    _has_setuptools = True
except ImportError:
    from distutils.core import setup

def check_dependencies():
    install_requires = []

    try:
        import propobject
    except ImportError:
        install_requires.append('propobject')
       
    try:
        import emcee
    except ImportError:
       install_requires.append('emcee')
       
     
    return install_requires

if __name__ == "__main__":

    install_requires = check_dependencies()

    if _has_setuptools:
        packages = find_packages()
        print(packages)
    else:
        # This should be updated if new submodules are added
        packages = ['mcsample']

    setup(name=DISTNAME,
          author=AUTHOR,
          author_email=MAINTAINER_EMAIL,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          install_requires=install_requires,
          packages=packages,
          package_data={},
          classifiers=[
              'Intended Audience :: Science/Research',
              'Programming Language :: Python :: 2.7',
              'Programming Language :: Python :: 3.5',              
              'License :: OSI Approved :: BSD License',
              'Topic :: Scientific/Engineering :: Astronomy',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS'],
      )
