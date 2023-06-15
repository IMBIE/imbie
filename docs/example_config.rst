Example Configuration
=====================

Below is an example copy of a configuration file that can be used by the IMBIE processor.::

    input_path "~/imbie/submissions-data/imbie_v2.02_data_20171011_GM"
    output_path "output/20200227/"
    plot_format "eps"
    export_data True
    use_dm True
    dmdt_window 3
    dmdt_method "weighted_least_squares"
    truncate_dmdt True
    truncate_avg False
    apply_dmdt_smoothing False
    reduce_window 1
    users_skip "mtalpe" "xpwujpl" "roelof" "IMBIE1" "jmouginot" "rignot2" "ahlstrom"
    combine_method "inv"
    group_avg_error_method "rms"
    sheet_avg_error_method "max"
    sum_errors_method "rss"
    table_format "html"
    bar_plot_min_time 2005
    bar_plot_max_time 2015
    plot_smooth_window 1.083333
    plot_smooth_iters 2
    imbie1_compare False