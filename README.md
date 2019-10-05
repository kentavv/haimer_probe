# haimer_probe
Use a Haimer 3D Taster probe and webcam as a LinuxCNC probe

Use machine vision to read the mechanical gauge dial face of a Haimer 3D probe and create an electronic probe for LinuxCNC.

A Microsoft LifeCam Cinema webcam is attached to a Haimer 3D Taser with a 3d printed part. The software monitors the two hands of the dial face, presenting debug views, and returning the combined value of the two dials. Keyboard commands begin edge probing and circle center locating.

Examples of use are shown at: https://www.youtube.com/watch?v=w3novypPqos

## Keyboard commands (haimer_camera.py)
* p: toggle display updates, image analysis continues
* r: toggle recording raw and displayed frames (overwriting)
* s: save current raw and displayed frames (non-overwriting)
* t: toggle tare calculations
* d: toggle debug view stages of image analysis
* q: quit
* KEY_LEFT: move pivot point to the left
* KEY_UP: move pivot point up
* KEY_RIGHT: move pivot point to the right
* KEY_DOWN: move pivot point down

## Keyboard commands (linuxcnc_driver.py)
These commands are in addition to those inherited from haimer_camera.py.

The additional commands are best seen on a numeric keypad. The direction shown on the keypad indicates the direction the probe approaches the part.

* 0: find center of hole, internal edges
* 4: find edge moving to the left, -x
* 6: find edge moving to the right, +x
* 8: find edge moving forward, away from operator, +y
* 2: find edge moving aft, towards the operator, -y
* 5: find top surface, moving down, towards part
* 1: find upper-right corner, moving to the left, then around the corner, and then towards the operator
* 3: find upper-left corner, moving to the right, then around the corner, and then towards the operator
* 7: find lower-right corner, moving to the left, then around the corner, and then away from the operator
* 9: find lower-left corner, moving to the right, then around the corner, and then away from the operator

## Some references
http://www.insticc.org/Primoris/Resources/PaperPdf.ashx?idPaper=73860
https://github.com/intel-iot-devkit/python-cv-samples/tree/master/examples/analog-gauge-reader
https://www.researchgate.net/publication/282582478_Machine_Vision_Based_Automatic_Detection_Method_of_Indicating_Values_of_a_Pointer_Gauge/fulltext/5686db1508ae051f9af42749/Machine-Vision-Based-Automatic-Detection-Method-of-Indicating-Values-of-a-Pointer-Gauge.pdf
https://www.degruyter.com/view/j/phys.2019.17.issue-1/phys-2019-0010/phys-2019-0010.xml
https://pdfs.semanticscholar.org/639e/d0c018925e6b900e6ddd2956b63ffd5f56dc.pdf
https://pdfs.semanticscholar.org/7083/1ea22f494e044c4861b3fb2bbfad578dd9a1.pdf
