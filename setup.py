import glob
import os 
from setuptools import setup, find_packages
#
# Begin setup
#
setup_keywords = dict()

setup_keywords['name']         = 'toolbox'
setup_keywords['packages']     = find_packages('py')
setup_keywords['package_dir']  = {'':'py'}
setup_keywords['install_requires'] = [
        'requests',
        'importlib-metadata; python_version<"3.10"',]
#
# set __version__ from version.py
#
with open("py/toolbox/version.py") as fp:
    exec(fp.read(),)
    setup_keywords['version']      = __version__ 

# 
# Treat everything in bin/ except *.rst as a script to be installed.
#
if os.path.isdir('bin'):
    setup_keywords['scripts'] = [fname for fname in glob.glob(os.path.join('bin', '*'))
        if not os.path.basename(fname).endswith('.rst')]

#
# Set other information 
#
setup_keywords['author']       = 'Yizhou Gu'
setup_keywords['author_email'] = 'guyizhou@sjtu.edu.cn'
setup_keywords['url'] = 'https://astroyzgu.github.io/'
setup_keywords['description'] = 'This is a toolbox for some simple tasks.'
setup_keywords['long_description'] = ''
if os.path.exists('README.md'):
    with open('README.md') as readme:
        setup_keywords['long_description'] = readme.read()
#-----------------------------------------------------------------
#
# Run setup command.
#
if __name__ == '__main__': 
    setup(**setup_keywords) 
