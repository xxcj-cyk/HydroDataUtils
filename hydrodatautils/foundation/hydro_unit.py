import xarray as xr
import re

def _convert_target_unit(target_unit):
    """Convert user-friendly unit to standard unit for internal calculations."""
    if match := re.match(r"mm/(\d+)(h|d)", target_unit):
        num, unit = match.groups()
        return int(num), unit
    return None, None


def streamflow_unit_conv(streamflow, area, target_unit="mm/d", inverse=True):
    """Convert the unit of streamflow data to mm/xx(time) for a basin or inverse.

    Parameters
    ----------
    streamflow: xarray.Dataset, numpy.ndarray, pandas.DataFrame/Series
        Streamflow data of each basin.
    area: xarray.Dataset or pint.Quantity wrapping numpy.ndarray, pandas.DataFrame/Series
        Area of each basin.
    target_unit: str
        The unit to convert to.
    inverse: bool
        If True, convert the unit to m^3/s.
        If False, convert the unit to mm/day or mm/h.

    Returns
    -------
    Converted data in the same type as the input streamflow.
    """
    # Convert the user input unit format
    num, unit = _convert_target_unit(target_unit)
    if unit:
        if unit == "h":
            standard_unit = "mm/h"
            conversion_factor = num
        elif unit == "d":
            standard_unit = "mm/d"
            conversion_factor = num
        else:
            raise ValueError(f"Unsupported unit: {unit}")
    else:
        standard_unit = target_unit
        conversion_factor = 1
    # Regular expression to match units with numbers
    custom_unit_pattern = re.compile(r"mm/(\d+)(h|d)")

    # Function to handle the conversion for numpy and pandas
    def np_pd_conversion(streamflow, area, target_unit, inverse, conversion_factor):
        if not inverse:
            result = (streamflow / area).to(target_unit) * conversion_factor
        else:
            result = (streamflow * area).to(target_unit) / conversion_factor
        return result.magnitude

    # Handle xarray
    if isinstance(streamflow, xr.Dataset) and isinstance(area, xr.Dataset):
        streamflow_units = streamflow[list(streamflow.keys())[0]].attrs.get(
            "units", None
        )
        if not inverse:
            if not (
                custom_unit_pattern.match(target_unit)
                or re.match(r"mm/(?!\d)", target_unit)
            ):
                raise ValueError(
                    "target_unit should be a valid unit like 'mm/d', 'mm/day', 'mm/h', 'mm/hour', 'mm/3h', 'mm/5d'"
                )

            q = streamflow.pint.quantify()
            a = area.pint.quantify()
            r = q[list(q.keys())[0]] / a[list(a.keys())[0]]
            # result = r.pint.to(target_unit).to_dataset(name=list(q.keys())[0])
            result = (r.pint.to(standard_unit) * conversion_factor).to_dataset(
                name=list(q.keys())[0]
            )
            # Manually set the unit attribute to the custom unit
            result_ = result.pint.dequantify()
            result_[list(result_.keys())[0]].attrs["units"] = target_unit
            return result_
        else:
            if streamflow_units:
                if custom_match := custom_unit_pattern.match(streamflow_units):
                    num, unit = custom_match.groups()
                    if unit == "h":
                        standard_unit = "mm/h"
                        conversion_factor = int(num)
                    elif unit == "d":
                        standard_unit = "mm/d"
                        conversion_factor = int(num)
                    # Convert custom unit to standard unit
                    r_ = streamflow / conversion_factor
                    r_[list(r_.keys())[0]].attrs["units"] = standard_unit
                    r = r_.pint.quantify()
                else:
                    r = streamflow.pint.quantify()
            else:
                r = streamflow.pint.quantify()
            if target_unit not in ["m^3/s", "m3/s"]:
                raise ValueError("target_unit should be 'm^3/s'")
            a = area.pint.quantify()
            q = r[list(r.keys())[0]] * a[list(a.keys())[0]]
            result = q.pint.to(target_unit).to_dataset(name=list(r.keys())[0])
            # dequantify to get normal xr_dataset
            return result.pint.dequantify()