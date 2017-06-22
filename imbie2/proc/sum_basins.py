from imbie2.model.collections import WorkingMassRateCollection
from imbie2.const.error_methods import ErrorMethod
from imbie2.const.basins import BasinGroup, ZwallyBasin, RignotBasin, IceSheet
from typing import Iterable


def sum_basins(rate_data: WorkingMassRateCollection, sheets: Iterable[IceSheet]=None) -> None:
    """
    given a WorkingMassRateCollection, finds users in the collection
    who have submitted a complete set of per-basin series for an ice sheet,
    but who have not provided a series for the complete ice sheet.

    For each such user found, it then accumulates the values of the contributing
    basins, to create a time-series for the complete ice sheet. These new series
    are added to the collection
    """
    if sheets is None:
        sheets = [IceSheet.eais, IceSheet.wais, IceSheet.apis, IceSheet.gris]

    users = list({s.user for s in rate_data})
    for user in users:
        user_data = rate_data.filter(user=user)
        for group, basin_set in zip([BasinGroup.zwally, BasinGroup.rignot], [ZwallyBasin, RignotBasin]):
            for sheet in sheets:
                basins = list(basin_set.sheet(sheet))
                sheet_data = user_data.filter(basin_id=basins)

                if user_data.filter(basin_id=sheet, basin_group=group):
                    continue

                if len(sheet_data) == len(basins):
                    series = sheet_data.sum(error_method=ErrorMethod.rss)

                    series.basin_id = sheet
                    series.basin_group = group
                    series.user = user
                    series.aggregated = True

                    rate_data.add_series(series)
