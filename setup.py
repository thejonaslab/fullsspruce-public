""" Package setup """
from setuptools import find_packages, setup
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup (
    name='FullSSPrUCe',
    version='0.0.1',
    author='The Jonas Lab',
    author_email='jonaslab@uchicago.edu',
    description='Full Spin System Predictions with Uncertainty',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/thejonaslab/nmr-forward',
    license='MIT',
    packages=[
        'fullsspruce', 'fullsspruce.model', 'fullsspruce.featurize'
    ],
    package_data = {'fullsspruce': ['default_predict_models/default_1H_model.meta',
                                    'default_predict_models/default_1H_model.chk',
                                    'default_predict_models/default_13C_model.meta',
                                    'default_predict_models/default_13C_model.chk',
                                    'default_predict_models/default_coupling_PT_model.meta',
                                    'default_predict_models/default_coupling_PT_model.chk',
                                    'default_predict_models/default_coupling_ETKDG_model.meta',
                                    'default_predict_models/default_coupling_ETKDG_model.chk']}, 
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    keywords=[
        'chemistry',
        'machine learning',
        'neural network',
        'full spin system',
        'nuclear magnetic resonance',
        'NMR'
    ],

    entry_points={
        'console_scripts': [
            'fullsspruce=fullsspruce.commandline_runner:predict',
    ],
    },    
)
