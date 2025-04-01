Grant Kirchhoff
2023.02.02

This is a guide to the code repository for Grant Kirchhoff's Deadtime Evaluation work for the NASA NSTGRO22 Fellowship Research. The relevant data is the data product from the PicoQuant boards used with the INPHAMIS backscatter lidar system in the ARSENL space. 

1) Plotting histograms from ".ARSENL" files
	
	"load_ARSENL_data.py"
	"plot_histogram.py"

	(A) To plot raw ".ARSENL" files as histograms, first set the filepath in "load_ARSENL_data.py", then run "plot_histogram.py". The user can set the time-window bounds using the "exclude" variable.

2) Running the deadtime vs. Poisson model evaluation script

	"evaluation_high_OD_iterate_vX.py"

	(A) When running the evaluation script, make sure that you are using netcdf data that has been generated via Matt Hayman's conversion code: "convert_timetag_csv". This is a critical step so make sure you have used "convert_timetag_csv" to generate the files. Contact Grant and/or Matt for access if this is not familiar to you or you do not have access yet. The file name should terminate with ".nc".

	(B) Now that you have the netcdf format data, make sure that each file name is prepended with the following: "ODXX", where "XX" is the OD value x 10. For example, "OD15" is OD1.5. This is injested in the code and used as the appropriate OD value for each file.

	(C) Near the beginning of the file, edit the "PARAMETERS" section according to what you would like for the run. Typical variables that are usually unchanged should be ["exclude_shots=True", "use_stop_idx=True", "run_full=True"]. During the polynomial fit optimization process, multiple polynomial orders are iterated through. You can set the polynomial order complexity via "M_min" and "M_max". 

	(D) IMPORTANT: Set the paths properly under "PATH VARIABLES". The save file names specify if the deadtime model was used, the number of laser shots included, the orders validated over, etc.

	(E) Once running, the script will loop through each file of datasets with different OD values, choose the optimal fit for each profile, then evaluate the fit against a high-fidelity dataset (i.e., minimal deadtime impact) (e.g., high-OD retrieval). The script will save relevant outputs (e.g., OD values, forward model, time vector) that can be used in future processing and are used in scripts like "plot_eval_loss.ipynb", for example.

	(F) To compare deadtime vs. Poisson model, make sure to run this whole sequence twice. The first with the 'include_deadtime' variable set to 'True' (deadtime model) and the second with it set to 'False' (Poisson model).

(3) Plotting evaluation loss results

	"plot_eval_loss.ipynb"

	(A) Make sure you can run iPython notebooks (e.g., Jupyter). Set 'load_dir' variable to directory path. This should be the same as the 'save_dir' variable path in "evaluation_high_OD_iterate_vX" you set. If you ran "evaluation_high_OD_iterate_vX" twice (for the Poisson and the deadtime models) then you should set the file names for their outputs as the 'df_dtime' and 'df_pois' variables e.g., 'eval_loss_dtimeTrue_order4-14_shots1.00E+05.csv' and 'eval_loss_dtimeFalse_order4-14_shots1.00E+05.csv'.

	(B) Run cells to generate plots.