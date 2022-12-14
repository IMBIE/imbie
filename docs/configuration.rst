Configuration Options
=====================

The IMBIE processor has a number of configuration options, which are specified using a configuration
file. This file should be a plain-text document. The configuration parameters, their purpose, and their
valid values are described below.

All values should be written as Python variables of the appropriate type – for example, text strings
should be contained by quote-marks, and numbers should be written without them. Each parameter
should be written in its own line of the file, and the name of the parameter should be the first entry in
the line, written without quote-marks.

Empty lines are ignored, and the parameters may be specified in any order. Some parameters are
optional, which means that it is not compulsory to provide an entry for them in the configuration. In the
case that an optional parameter is not provided, the default value/behaviour is described here.
The list of options are:

* `input_path` – The directory to search for input data. Absolute or relative paths can be used.
  The processor will search all subdirectories of the input path for ‘.answers.json’ files which
  contain details of data submissions. The processor will then read any CSV-format data files
  specified in the relevant fields of the JSON document

* `output_path` – The directory in which to save plots and tables. Absolute or relative paths
  can be used.

* `export_data` – Optional field. A Boolean value (True or False). If True, the processor will
  export the computed data as CSV files. By default, the value is considered to be False.

* `plot_format` – Optional field. Specifies the format in which to save plots – should be one of
  "png", "jpg", "svg", or "pdf". If this parameter is omitted, the plots will not be saved and will
  instead be rendered in a window.

* `start_date` – Optional field. Specifies the date (in decimal years) from which to begin the
  analysis. All time-series starting before the specified date will be cropped to begin at this date
  (or omitted, if ending before the date provided).

* `stop_date` – Optional field. Specifies the date (in decimal years) at which to end the analysis.
  All time-series ending after the specified date will be cropped to end at this date (or omitted, if
  starting before the date provided).

* `align_date` – Optional field. Specifies the date (in decimal years) at which to align the
  integrated time-series. If this parameter is absent, then the time-series are instead offset such
  that their start-points are aligned with the average time-series.

* `combine_method` – Optional field. Specifies the method used to combine multiple time-
  series. This must be one of:

* `"eqg"` – Equally-weighted groups: Each expirment group has an equal contribution to
  the overall average. By default, this method is used.

* `"eqs"` – Equally-weighted series: Each individual contribution has an equal
  contribution to the overall average

* `"inv"` – Inverse error-weighted: Each individual contribution is weighted according to
  the inverse of its error margin.

* `"imbie1"` – A special method designed to replicate the behaviour of the processor
  used in the IMBIE 2012 analysis: averages are calculated with the same method used
  when averaging time series in the IMBIE 2012 analysis. Groups are equally weighted,
  and error margins are RMS over square root of the number of elements.

* `group_avg_error_method` – Optional field. Specifies the method that should be used to
  compute the error margin when multiple dM/dt time-series from the same experiment group
  are averaged together to produce a single estimate for the group. If omitted, the default
  behaviour depends on the method selected in combine_method. The value must be one of:
  
  * `"sum"` – The sum of the errors.
  
  * `"rms"` – Root Mean Squared.
  
  * `"rss"` – Root Sum Squared.
  
  * `"avg"` – The mean.

  * `"imbie1"` – method used by the IMBIE 2012 analysis: errors are calculated with the
    same method used when averaging time series in the IMBIE 2012 analysis, they are
    RMS over square root of the number of elements.

* `sheet_avg_error_method` – Optional field. Specifies the method that should be used to
  compute the error margin when multiple dM/dt time-series from different experiment groups
  are averaged together to produce a single estimate for an ice sheet. If omitted, the default
  behaviour depends on the method selected in combine_method. The value must be one of:

  * `"sum"` – The sum of the errors.
  
  * `"rms"` – Root Mean Squared.
  
  * `"rss"` – Root Sum Squared.
  
  * `"avg"` – The mean.
  
  * `"imbie1"` – method used by the IMBIE 2012 analysis: errors are calculated with the
    same method used when averaging time series in the IMBIE 2012 analysis, they are
    RMS over square root of the number of elements.

* `sum_errors_method` – Optional field. Specifies the method that should be used to compute
  the error margin when multiple time-series are summed together. This must be one of:

  * `"sum"` – The sum of the errors. By default, this method is used.
  
  * `"rms"` – Root Mean Squared.
  
  * `"rss"` – Root Sum Squared.
  
  * `"avg"` – The mean.
  
  * `"imbie1"` – method used by the IMBIE 2012 analysis: errors are calculated with the
    same method used when averaging time series in the IMBIE 2012 analysis, they are
    RMS over square root of the number of elements.

* `average_nsigma` – Optional field. Specifies the maximum margin when computing the
  average of multiple time-series. Values beyond this multiple of the standard deviation from the
  mean are considered to be outliers, and omitted from the average. By default, there is no
  maximum margin and all values will contribute to the average.

* `users_skip` – Optional field. A list of contributions (specified by the contributer’s username,
  with all capital letters, accents or diacritic marks removed, eg 'Sørensen' should be written
  'sorensen') to exclude from the analysis. Multiple usernames can be specified, separated by whitespace.

* `users_mark` – Optional field. A list of contributions (specified by the contributer’s surname)
  to mark in dM/dt and dM time-series plots. Multiple usernames can be specified, separated by
  whitespace. This parameter can be used to indicate the identity of outlying contributions.

* `plot_smooth_window` – Optional field. Specifies the time-window (in decimal years) which
  should be used when applying a moving average to dM/dt time-series plots. By default, no
  moving average is applied.

* `plot_smooth_iters` – Optional field. Specfies the number of iteration of smoothing to apply to
  plotted series. Default is 1 if omitted.

* `bar_plot_min_time` – Optional field. Specifies the minimum date from which the mean and
  standard deviation dM/dt are calculated for the error-bar plot. By default, there is no minimum
  date.

* `bar_plot_max_time` – Optional field. Specifies the maximum date from which the mean
  and standard deviation dM/dt are calculated for the error-bar plot. By default, there is no
  maximum date.

* `include_la` – Optional field. A Boolean value (True or False). If True, an additional "LA"
  (Laser Altimetry) expirement group will be considered by the processor. If the parameter is
  omitted, the value is considered to be False.

* `methods_skip` – Optional field. A list of experiment groups to exclude from the analysis.
  Multiple groups can be specified, separated by whitespace. Valid values are:

  * `"RA"`: The Altimetry group
  
  * `"GMB"`: The Gravimetry group
  
  * `"IOM"`: The Mass-Budget group
  
* `use_dm` – Optional field. Enables reading dM contributions in order to convert these to dM/dt
  data. False by default if omitted.

* `dmdt_window` – Optional field. Sets the length (in decimal years) of the curve-fitting window
  used for dM-to-dM/dt conversion. If the parameter is omitted, the value is considered be 1
  year.

* `dmdt_method` – Optional field. Specifies the method to be used for the curve-fitting in the dM-
  to-dM/dt conversion. Valid settings are:

  * `"ordinary_least_squares"` – basic fitting (default if parameter is omitted)
  
  * `"weighted_least_squares"` – inverse-error weighted least squares fitting

* `truncate_dmdt` – Optional field. Sets whether or not dM/dt series produced by the dM-to-
  dM/dt conversion process should be cropped to the length within which a complete window
  can be constructed from the input dM data. True by default if omitted.

* `truncate_avg` – Optional field. Toggles whether group average series should be truncated
  to the length of contributions when truncate_dmdt is applied. False by default.

* `apply_dmdt_smoothing` – Optional field. Specifies if the dM/dt contributions should be
  smoothed after reading. The window used for this smoothing is the same as the value of
  dmdt_window

* `reduce_window` – sets the width (in decimal years) of the window over which to apply a
  moving average on the contributions, reducing the number of data points in each series. If
  omitted, the averaging is not applied.

* `data_smoothing_window` – Optional field. Specifies the width (in decimal years) of
  windowed smoothing to apply to internal data. If omitted, no smoothing is applied.

* `data_smoothing_iters` – Optional field. Specfies the number of iteration of smoothing to
  apply to internal data series. Default is 1 if omitted.

* `export_smoothing_window` – Optional field. Specifies the width (in decimal years) of
  windowed smoothing to apply to exported data. If omitted, no smoothing is applied.

* `export_smoothing_iters` – Optional field. Specfies the number of iteration of smoothing
  to apply to exported data series. Default is 1 if omitted.

* `imbie1_compare` – Optional field. Toggles whether to provide a plotted comparison with
  IMBIE-1 data. True by default.

* `output_timestep` – Optional field. Sets the interval between data points in output files (in
  decimal years). If no value is provided, the data will not be adjusted.

* `output_offset` – Optional field. Sets the faction of the year at which the first data point in
  the output files should be provided. All subsequent points will be spaced according the value
  of output_timestep. If omitted, no adjustment is performed.

* `smb_data` – Specifies the path of Surface Mass Balance CSV data to use for calculating ice
  sheet dynamics for Greenland

* `data_min_time` – Optional field. In conjunction with data_max_time, sets a time window to
  be applied to the input data when read.

* `data_max_time` – Optional field. See data_min_time

* `dmdt_tapering` – Optional field. Boolean value, when True, applies window tapering method
  to dm-to-dmdt conversion. Default False.

* `dmdt_monthly` – Optional field. Forces monthly interpolation of data points when performing
  dm-to-dmdt conversion when set to True. Default False.

The default axis limits can be changed for certain plots only, using the following six options.
This affects only plots whose names start with group_rate_boxes, groups_mass_intercomparison,
groups_rate_intercomparison, regions_mass_intercomparison and regions_rate_intercomparison.

* `plotter_min_time` – Optional field. Sets earliest date in plot time range. Default if omitted is 1990.

* `plotter_max_time` – Optional field. Sets latest date in plot time range. Default if omitted is 2022.

* `plotter_min_dmdt` – Optional field. Sets lowest value in plot dm/dt range. Default if omitted is ``-``500 (Gt/yr).

* `plotter_max_dmdt` – Optional field. Sets highest value in plot dm/dt range. Default if omitted is 200 (Gt/yr)

* `plotter_min_dm` – Optional field. Sets lowest value in plot dm range. Default if omitted is ``-``9000 (Gt)

* `plotter_max_dm` – Optional field. Sets highest value in plot dm range. Default if omitted is 3000 (Gt)


  

Example configuration file - used for IMBIE 3
=============================================

input_path "/home/xxx/imbie_2022_analysis/data/submissions/"

output_path "/home/xxx/imbie_2022_analysis/outputs/imbie3/"

plot_format "png"


data_min_time 1971

data_max_time 2023


export_data True


use_dm True

dmdt_window 3

dmdt_method "weighted_least_squares"

truncate_dmdt False

truncate_avg False

apply_dmdt_smoothing True


reduce_window 1


combine_method "inv"

group_avg_error_method "rms"

sheet_avg_error_method "max"

sum_errors_method "rss"

table_format "html"

bar_plot_min_time 1971

bar_plot_max_time 2023

plot_smooth_window 1.083333

plot_smooth_iters 2


imbie1_compare False


dmdt_monthly True

dmdt_tapering True


plotter_min_time 1971.0

plotter_max_time 2023.0

plotter_min_dmdt -600

plotter_max_dmdt 300

plotter_min_dm -10000

plotter_max_dm 4000
