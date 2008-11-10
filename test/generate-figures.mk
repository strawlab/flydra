generated_figures/plot_kalman_2d.png:
	mkdir -p generated_figures
	flydra_analysis_plot_kalman_2d flydra/a2/sample_datafile.h5 --save-fig=$@
