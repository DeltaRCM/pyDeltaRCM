import numpy as np

import pyDeltaRCM

class MassConservationCheck(pyDeltaRCM.DeltaModel):

    def __init__(self, input_file, **kwargs):

        super().__init__(input_file, **kwargs)

        self.inlets_flat = self.inlet

        self.cumulative_input_volume = 0
        self.cumulative_lost_volume = 0
        self.cumulative_exported_volume = 0

        print(f"|" + 66*"-" + "|")
        print(f"| vol input     | vol deposit   | % err    | % lost   | % exported | ")
        print(f"|" + 66*"-" + "|")

    def hook_after_finalize_timestep(self):
        self._verify_mass_conservation()

    def _verify_mass_conservation(self):
        """
        On each timestep, verify the conservation of mass

        DeltaRCM is never mass conservative of water, though we should verify
        the correct amount of water is entering at the inlet cells.

        Sediment mass should be conserved. We should verify the correct amount
        of sediment is entering each inlet cell. We should also verify that the
        total amount of landscape change (volume) is close to the total
        amount of sediment put into the domain.
        """

        # check the change in volume of the deposit for this timestep
        act_deposit_volume = np.sum((self.eta - self.eta_init) * self.dx**2)
        input_sed_volume = self.Qs0 * self.dt

        # now check it accounting for whatever was lost or exported
        exported_volume = self._sr.Vp_exported
        lost_volume = self._sr.Vp_lost
        
        self.cumulative_lost_volume += lost_volume
        self.cumulative_exported_volume += exported_volume
        self.cumulative_input_volume += input_sed_volume

        cumulative_lost_frac_error = self.cumulative_lost_volume / self.cumulative_input_volume
        cumulative_exported_frac_error = self.cumulative_exported_volume / self.cumulative_input_volume

        deposit_volume_error = (act_deposit_volume - self.cumulative_input_volume) / self.cumulative_input_volume

        print(f"|{self.cumulative_input_volume:>15.1e}|{act_deposit_volume:>15.2e}|{deposit_volume_error*100:>10.2f}|{cumulative_lost_frac_error*100:>10.2f}|{cumulative_exported_frac_error*100:>12.2f}| ")

    def finalize(self):
        super().finalize()
        print(f"|" + 66*"-" + "|")
        print("")


if __name__ == "__main__":
    
    demo_conditions = dict(f_bedload=1, Np_sed=200, itermax=1)
    # normal conditions yield good balance
    print("A good balance yields:")
    model = MassConservationCheck(input_file=None, **demo_conditions)
    for _ in np.arange(10):
        model.update()
    model.finalize()

    # too small of stepmax causes many "lost" partciles
    print("Too small of stepmax yields:")
    model = MassConservationCheck(input_file=None, **demo_conditions, stepmax=5)
    for _ in np.arange(10):
        model.update()
    model.finalize()

    # too small of domain causes many "exported" partciles
    print("Too small of domain yields:")
    model = MassConservationCheck(input_file=None, **demo_conditions, Length=500, Width=1000)
    for _ in np.arange(10):
        model.update()
    model.finalize()
