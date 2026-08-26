"""MuJoCo LQR controller entry point.

The former file was a Webots controller and imported five unavailable Webots
project modules.  The active wheeled-leg model is MuJoCo, so this launcher
keeps the original command name while using the deployed MuJoCo LQR loop.
"""

from lqr_deploy import main


if __name__ == "__main__":
    main()
