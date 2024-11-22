from setuptools import setup, find_packages

setup(
    name='probabilistic_urban_evolution',  # Replace with your package name
    version='0.1.0',  # Replace with your package version
    packages=find_packages(),  # Automatically find packages
    install_requires=[ 'pandas', 'geopandas', 'numpy', 'seaborn', 'matplotlib', 'plotly', 'hmmlearn', 'scikit-gstat', 'scikit-learn'],
)


