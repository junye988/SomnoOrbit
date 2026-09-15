"""Train CNN--BiGRU and construct SomnoOrbit for the HMC sleep dataset."""

from model.training import run_demo


if __name__ == "__main__":
    run_demo("hmc")
