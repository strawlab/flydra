from setuptools import setup
import flydra.version

version = flydra.version.__version__

setup(name='flydra',
      version=version,
      author='Andrew Straw',
      author_email='strawman@astraw.com',
      description='multi-headed fly-tracking beast',
      packages = ['flydra.a3','flydra.sge_utils'],
      entry_points = {
    'console_scripts': [

# SGE/flydra.astraw.com commands
    'flydra_sge_download_jobs = flydra.sge_utils.sge_download_jobs:main',
    'flydra_sge_run_job = flydra.sge_utils.sge_run_job:main',
    ],
    }
      )
