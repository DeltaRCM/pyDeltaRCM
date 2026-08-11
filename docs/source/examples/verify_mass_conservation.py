import numpy as np

import pyDeltaRCM


class MassConservationCheck(pyDeltaRCM.DeltaModel):

    def __init__(self, input_file, **kwargs):

        super().__init__(input_file, **kwargs)

        self.inlets_flat = self.inlet

        # trackers for cumulative volume changes
        self.cumulative_input_volume = 0
        self.cumulative_lost_volume = 0
        self.cumulative_exported_volume = 0
        self.cumulative_inlet_volume = 0

        # some code to help format a nice table
        self._fields = [
            f"timestep",
            f"vol input",
            f"vol deposit",
            f"% err",
            f"% lost",
            f"% exported",
            f"% at inlet",
        ]
        self._sty0 = ["g", ".1e", ".1e", ".2f", ".2f", ".2f", ".2f"]
        self._fields_len = [
            np.maximum(len(s), 8) for s in self._fields
        ]  # field plus two spaces on each side
        self._tot_len = (
            np.sum(self._fields_len)
            + 2
            + (len(self._fields) - 1)
            + 2
            + 2 * (len(self._fields) - 1)
        )  # fields + end |s + middle |s + end spcs + middle spcs

        # print table header
        self._format_table()
        self._format_table(self._fields)

    def _format_table(self, entry=None):

        if entry is None:
            print(f"|" + ((self._tot_len - 2) * "-") + "|")
        else:
            fl = self._fields_len
            sty0 = self._sty0
            # make list of stings
            row = []
            for i in np.arange(len(entry)):
                styi = sty0[i] if not isinstance(entry[i], str) else ""
                row.append(f"{entry[i]:>{fl[i]}{styi}}")
            print("| " + " | ".join(row) + " |")

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
        act_deposit_volume = np.sum((self.eta - self.eta_init) * self.dx**2) * (
            1 - self.porosity
        )
        input_sed_volume = self.Qs0 * self.dt
        act_deposit_volume = np.sum((self.eta - self.eta_init) * self.dx**2)

        # changes in input, lost, exported, inlet
        lost_volume = self._sr.Vp_lost + self._mr.Vp_lost
        exported_volume = self._sr.Vp_exported + self._mr.Vp_exported
        inlet_volume = self._Vp_inletbc

        # accumulate
        self.cumulative_input_volume += input_sed_volume
        self.cumulative_lost_volume += lost_volume
        self.cumulative_exported_volume += exported_volume
        self.cumulative_inlet_volume += inlet_volume

        cumulative_lost_frac_error = (
            self.cumulative_lost_volume / self.cumulative_input_volume
        )
        cumulative_exported_frac_error = (
            self.cumulative_exported_volume / self.cumulative_input_volume
        )
        cumulative_inlet_frac_error = (
            self.cumulative_inlet_volume / self.cumulative_input_volume
        )
        cumulative_deposit_frac_error = (
            act_deposit_volume - self.cumulative_input_volume
        ) / self.cumulative_input_volume

        entry = [
            self._time_iter,
            self.cumulative_input_volume,
            act_deposit_volume,
            cumulative_deposit_frac_error * 100,
            cumulative_lost_frac_error * 100,
            cumulative_exported_frac_error * 100,
            cumulative_inlet_frac_error * 100,
        ]
        self._format_table(entry)

    def finalize(self):
        super().finalize()  # .finalize() from base model
        # print table footer
        self._format_table()
        print("")  # blankline


if __name__ == "__main__":

    demo_conditions = dict(Np_sed=200, itermax=1)
    # normal conditions yield good balance
    print("Default parameters yield cumulative volume conservation errors (%):")
    model = MassConservationCheck(input_file=None, **demo_conditions)
    for _ in np.arange(10):
        model.update()
    model.finalize()

    # too small of stepmax causes many "lost" partciles
    print("Too small of stepmax yields cumulative volume conservation errors (%):")
    model = MassConservationCheck(input_file=None, **demo_conditions, stepmax=5)
    for _ in np.arange(10):
        model.update()
    model.finalize()

    # too small of domain causes many "exported" partciles
    print("Too small of domain yields cumulative volume conservation errors (%):")
    model = MassConservationCheck(
        input_file=None, **demo_conditions, Length=500, Width=1000
    )
    for _ in np.arange(10):
        model.update()
    model.finalize()
