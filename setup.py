from setuptools import setup

setup(
    name="ghosts",
    author="Johan Bregeon",
    author_email="bregeon@in2p3.fr",
    url = "https://github.com/bregeon/ghosts",
    packages=["ghosts"],
    description="Analysis of Rubin telescope ghost images",
    setup_requires=['setuptools_scm'],
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    use_scm_version={"write_to":"ghosts/_version.py"},
    include_package_data=True,
    classifiers=[
        "Development Status :: 1 - Beta",
        "License :: OSI Approved :: GPL License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        ],
    install_requires=[#"batoid",
                      "python>=3.9",
#                      "matplotlib",
#                      "scipy",
#                      "pandas",
                      "openpyxl",
                      "setuptools_scm"]
)
