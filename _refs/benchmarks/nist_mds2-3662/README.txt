NIST Data Publication:
Dataset for Model Validation of Transient Melt Pool Dynamics
in Laser Powder Bed Fusion of Nickel Super Alloy 625: Top Surface Melt Pool Width 
and Area Measurements from Beam on Plate Experiments

Version 1.0.0
DOI: https://doi.org/10.18434/mds2-3662

Authors:
  Jesse  Redford
    National Institute of Standards and Technology
  Vijaya  Holla
    Technical University of Munich

Contact:
  Jesse Redford
    jesse.redford@nist.gov

Description:

The files include experimental data from the publication titled "The trace of
heat: on the predictive power of modeling transient diffusion"
(https://doi.org/10.1007/s40964-025-01147-9). This data was used to develop
assess the validity of the transient heat equation with phase change and
temperature-dependent coefficients as a model to predict the evolution of melt
pools for rapid turnaround scan strategies in a laser powder bed fusion (PBF-LB)
additive manufacturing (AM) process. The dataset contains the scan paths and
parameters used to carry out a series of PBF-LB beam on plate experiments with 
top surface bright field images and tabulated measurements of the melt pool width
and top surface area of the last track of each fabricated sample.

----------
References
----------

This collection is a supplement to:
  Holla, V., Redford, J., Kopp, P., & Kollmannsberger, S. (2025). The trace of  
heat: on the predictive power of modeling transient diffusion. Progress in  
Additive Manufacturing. https://doi.org/10.1007/s40964-025-01147-9

--------------
Data Use Notes
--------------

This data is publicly available according to the NIST statements of
copyright, fair use and licensing; see
https://www.nist.gov/director/copyright-fair-use-and-licensing-statements-srd-data-and-software

You may cite the use of this data as follows:
Redford, Jesse, Holla, Vijaya (2025), Dataset for model validation of transient melt pool dynamics
in laser powder bed fusion of nickel super alloy 625: top surface melt pool width 
and area measurements from beam on plate experiments,
Version 1.0.0, National Institute of Standards and Technology,
https://doi.org/10.18434/mds2-3662 (Accessed: [give download date])


--------------
Disclaimer
--------------

This data/work was created by employees of the National Institute of Standards and Technology (NIST), an agency of the Federal Government. Pursuant to title 17 United States Code Section 105, works of NIST employees are not subject to copyright protection in the United States.  This data/work may be subject to foreign copyright.
The data/work is provided by NIST as a public service and is expressly provided “AS IS.” NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED OR STATUTORY, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST does not warrant or make any representations regarding the use of the data or the results thereof, including but not limited to the correctness, accuracy, reliability or usefulness of the data. NIST SHALL NOT BE LIABLE AND YOU HEREBY RELEASE NIST FROM LIABILITY FOR ANY INDIRECT, CONSEQUENTIAL, SPECIAL, OR INCIDENTAL DAMAGES (INCLUDING DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, AND THE LIKE), WHETHER ARISING IN TORT, CONTRACT, OR OTHERWISE, ARISING FROM OR RELATING TO THE DATA (OR THE USE OF OR INABILITY TO USE THIS DATA), EVEN IF NIST HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

To the extent that NIST may hold copyright in countries other than the United States, you are hereby granted the non-exclusive irrevocable and unconditional right to print, publish, prepare derivative works and distribute the NIST data, in any medium, or authorize others to do so on your behalf, on a royalty-free basis throughout the world.

You may improve, modify, and create derivative works of the data or any portion of the data, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the data and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the data:  Data citation recommendations are provided at https://www.nist.gov/open/license.
Permission to use this data is contingent upon your acceptance of the terms of this agreement and upon your providing appropriate acknowledgments of NIST’s creation of the data/work.

Certain commercial equipment, instruments, or materials are identified in this paper in order to specify the experimental procedure adequately.  Such identification is not intended to imply recommendation or endorsement by NIST, nor is it intended to imply that the materials or equipment identified are necessarily the best available for the purpose.

---------------
Version History
---------------

1.0.0 (this version)
  initial release


--------------------------
Methodological Information
--------------------------

Machine settings and materials:
The experiments were conducted using an EOS M290 machine located at the NIST Gaithersburg campus. Beam-on-plate experiments were performed on a 76 mm x 76 mm x 6 mm nickel alloy 625 plate using vendor-recommended parameters for the alloy. These parameters included a laser power of 285 W, a scan speed of 960 mm/s, and a hatch spacing of 110 μm. The skywriting option was enabled, which turned off the laser after reaching the end of a track, allowing the beam to decelerate and accelerate back to the programmed velocity before turning back on at the start of the next track.

Scanning strategy data:
This data represents an 18-track rapid-turnaround artifact with an isosceles trapezoid geometry, captured at various stages of fabrication. The scanning strategy was recorded using a Printrite 3D system by Sigma Additive, which samples commands sent to the EOS M290 controller at 100 kHz. The artifact was produced as a set of partial geometries, each following the entire 18-track scan strategy up to a specific track number (1 to 18) and stops prematurely at that number to create solidified "snapshots" of the fabrication process. For the converging case, the first track started at the wide end of the trapezoid, while for the diverging case, it started at the narrow end. The scanning strategy data is provided in CSV files for both converging and diverging 18-track samples, containing x, y, laser power, and gating (laser on/off) commands following the xy2-100 protocol. To obtain the scan strategy for a specific partial geometry to input into a simulation model the CSV file can be truncated to the corresponding number of rows.

Melt Pool measurments:
Top-surface bright-field images of each fabricated sample were collected using a Zeiss AxioImager.Z2 microscope.  Two operators used ImageJ software to trace the inner chevron boundaries of the solidified melt pool of the last fabricated track of each sample, the maximum melt pool width and total top-surface area of the melt pool for each sample was then tabulated. 

Additional information:
For more details and diagrams, refer to the original publication: https://doi.org/10.1007/s40964-025-01147-9.


-------------
Data Overview
-------------

Files included in this publication:

Scan Strategy Data

    singleTrack.csv: Scan strategy for a single track scan

    scanStrategyDiverging.csv:  Scan strategy for the diverging case

    scanStrategyConverging.csv: Scan strategy for the converging case

Image Data

    Set1_single_track.bmp: One single-track image

    Set1_Diverging_(2,3,4,5,6,7,8,18)tracks.tif: Nine diverging scan images

    Set1_Converging_18tracks.tif: One converging scan image

    Set1_single_track_withlines.bmp: One single-track image approximated melt pool width line in red color 

    Set1_Diverging_(2,3,4,5,6,7,8,18)tracks_withlines.tif: Nine diverging scan images with approximated melt pool lines in red color

    Set1_Converging_18tracks_withlines.tif: One converging scan image approximated melt pool lines in red color 

    Set2_Diverging_Samples.png: Stitched image containing 1 to 18 track samples with three repeats for diverging case

    Set2_Converging_Samples.png: Stitched image containing 1 to 18 track sample with three repeats for converging case

Measurement Data

    Measurements.xlsx:  Contains tabulated measurements of the melt pool width and area for images in Set1 and Set2
