# evolutionary
Evolutionary algorithms for SNNs.

## Install
Some packages come directly from repositories and need installing via `pip` directly (instead of `setuptools`). So run these commands:

```bash
$ git clone https://github.com/Huizerd/evolutionary
$ pip install -e evolutionary/
$ pip install -r evolutionary/requirements.txt
```

Code is formatted with [Black](https://github.com/psf/black) using a pre-commit hook. To configure it, run:

```bash
$ pre-commit install
```

Also note that `gcc`, `g++`, `python3-dev` and `build-essential` are needed to compile C/C++ extensions that speed up DEAP, so make sure these are installed on your system. Also, install `python3-matplotlib` to be able to plot 3D figures.

#### Solutions to known issues

- OSError: [Errno 24] Too many open files

Modify `/etc/systemd/user.conf` and `/etc/systemd/system.conf` with the following line (this takes care of graphical login):

```
DefaultLimitNOFILE=65535
```

Modify `/etc/security/limits.conf` with the following lines (this takes care of non-GUI login):

```
* hard nofile 65535
* soft nofile 65535
```

where `*` can be replaced with the username. Reboot your computer for changes to take effect.
