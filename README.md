# ghosts

Analyzing ghosts from the CCOB Narrow beam and beyond

Read documentation on ReadTheDocs: [latest](https://ghosts.readthedocs.io/en/latest/ghosts.html)

## Install
### conda, then pip
```
> conda env create -f environment.yml
> conda activate combined_fit
> git clone https://github.com/bregeon/ghosts.git
> cd ghosts
> pip install -r requirements.txt
> pip install -e .
```

### pip only
```
> conda create -n my_ghosts_env python=3.9
> pip install -r requirements.txt
> git clone https://github.com/bregeon/ghosts.git
> cd ghosts
> pip install -e .
```

## People

* [Johan Bregeon](https://github.com/bregeon) (CNRS IN2P3 LPSC)


## License, Contributing etc

The code in this repository is available for re-use under the GPL v3 license.

See code coverage at: [codecov](https://app.codecov.io/gh/bregeon/ghosts)