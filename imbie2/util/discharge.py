import numpy as np

from imbie2.model.series import WorkingMassRateDataSeries
from imbie2.util.functions import match, ts2m


def calculate_discharge(mass_balance: WorkingMassRateDataSeries, surface_mass_balance: WorkingMassRateDataSeries) -> WorkingMassRateDataSeries:
    t_smb, dmdt_smb = ts2m(surface_mass_balance.t, surface_mass_balance.dmdt)
    _, errs_smb = ts2m(surface_mass_balance.t, surface_mass_balance.errs)

    t_mb, dmdt_mb = ts2m(mass_balance.t, mass_balance.dmdt)
    _, errs_mb = ts2m(mass_balance.t, mass_balance.errs)

    i_smb, i_mb = match(t_smb, t_mb, epsilon=1./24.)

    discharge_t = t_smb[i_smb]
    discharge_rate = dmdt_mb[i_mb] - dmdt_smb[i_smb]
    discharge_errs = np.sqrt(
        errs_mb[i_mb] ** 2 + errs_smb[i_smb] ** 2
    )

    return WorkingMassRateDataSeries(
        mass_balance.user, mass_balance.user_group, mass_balance.data_group, mass_balance.basin_group,
        mass_balance.basin_id, mass_balance.basin_area, discharge_t, mass_balance.a,
        discharge_rate, discharge_errs, mass_balance.computed, mass_balance.merged, 
        mass_balance.aggregated, mass_balance.contributions, mass_balance.trunc_extent
    )