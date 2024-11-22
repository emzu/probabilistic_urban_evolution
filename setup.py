from setuptools import setup, find_packages

setup(
    name='probabilistic-urban-evolution',  # Replace with your package name
    version='0.1.0',  # Replace with your package version
    packages=find_packages(),  # Automatically find packages
    install_requires=[ 'os', 'pandas', 'geopandas', 'numpy', 'seaborn', 'matplotlib', 'plotly', 'hmmlearn', 'scikit-gstat', 'sklearn'],
)


